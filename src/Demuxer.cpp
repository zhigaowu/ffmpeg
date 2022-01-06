
#pragma warning (disable:4819)

#include "Demuxer.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavutil/avutil.h>
#ifdef __cplusplus
}
#endif

#include <sstream>
#include <chrono>

namespace ffmpeg 
{
    inline static int64_t milliseconds_since_epoch()
    {
        typedef std::chrono::milliseconds time_precision;
        typedef std::chrono::time_point<std::chrono::system_clock, time_precision> time_point_precision;
        time_point_precision tp = std::chrono::time_point_cast<time_precision>(std::chrono::system_clock::now());
        return tp.time_since_epoch().count();
    }

    inline static int InterruptCb(void* userdata)
    {
        Demuxer* demuxer = static_cast<Demuxer*>(userdata);
        if (demuxer->timeOut())
        {
            return 1;
        }
        return 0;
    }

    Demuxer::Demuxer()
        : format_context(nullptr), streams(nullptr)
        , seek_pts(0)
        , video_stream_index(-1), video_codecpar(nullptr), frame_rate(25.0)
        , audio_stream_index(-1), audio_codecpar(nullptr)
        , _read_timeout(2000LL), _read_at(0LL)
    {
    }

    Demuxer::~Demuxer()
    {
    }

    int Demuxer::Open(int64_t connect_timeout, int64_t read_timeout, const std::map<std::string, std::string>& options, int64_t seek_micros)
    {
        int res = 0;

        AVDictionary* param = nullptr;
        do 
        {
            if (!(format_context = avformat_alloc_context()))
            {
				res = AVERROR(ENOMEM);
                break;
            }

            for (std::map<std::string, std::string>::const_iterator optr = options.cbegin(); optr != options.cend() && res >= 0; ++optr)
            {
                res = av_dict_set(&param, optr->first.c_str(), optr->second.c_str(), 0);
            }
            if (res < 0)
            {
                break;
            }

            // set connection timeout
            _read_timeout = connect_timeout;
            _read_at = milliseconds_since_epoch();
            format_context->interrupt_callback = AVIOInterruptCB{ InterruptCb, this };

            const AVInputFormat* format = nullptr;
            if ((res = open(&format, &param)) < 0)
            {
                break;
            }

            if ((res = avformat_find_stream_info(format_context, nullptr)) < 0)
            {
                break;
            }

            // update read timeout
            _read_timeout = read_timeout;

            // parse common used parameters
            parseParameters();

            // seek frame if format is seekable
            Seek(seek_micros);
        } while (false);

        if (param)
        {
            av_dict_free(&param);
        }

        return res;
    }

    int Demuxer::Seek(int64_t seek_micros)
    {
        int res = 0;

        if (seek_micros <= 0)
        {
            seek_micros = 0;
        }

        // seek on video if possible
        int stream_index = video_stream_index;
        if (video_stream_index < 0)
        {
            stream_index = audio_stream_index;
        }

        const AVStream* stream = format_context->streams[stream_index];
        seek_pts = av_rescale_q(seek_micros, AVRational{ 1, AV_TIME_BASE }, stream->time_base);
        if (seek_pts > stream->duration || (res = av_seek_frame(format_context, stream_index, seek_pts, AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_FRAME)) < 0)
        {
            seek_pts = 0;
        }

        return res;
    }

    int Demuxer::Read(AVPacket** packet)
    {
        _read_at = milliseconds_since_epoch();
        int res = av_read_frame(format_context, *packet);
        if (res >= 0)
        {
            (*packet)->time_base = streams[(*packet)->stream_index]->time_base;
            (*packet)->opaque = (void*)streams[(*packet)->stream_index]->codecpar->codec_type;
        }
        return res;
    }

	bool Demuxer::Local()
	{
		if (video_stream_index >= 0)
		{
			return streams[video_stream_index]->nb_frames > 0;
		}
		return false;
	}

	bool Demuxer::timeOut()
    {
        return _read_at + _read_timeout <= milliseconds_since_epoch();
    }

    void Demuxer::Close()
    {
        if (format_context)
        {
            avformat_close_input(&format_context);
            format_context = nullptr;
        }
    }

    void Demuxer::parseParameters()
    {
        streams = format_context->streams;
        for (int i = 0; i < static_cast<int>(format_context->nb_streams); ++i)
        {
            const AVCodecParameters *codecpar = streams[i]->codecpar;
            switch (codecpar->codec_type)
            {
            case AVMEDIA_TYPE_VIDEO:
            {
                video_stream_index = i;
                video_codecpar = codecpar;
                if (streams[i]->avg_frame_rate.den > 0)
				{
					frame_rate = av_q2d(streams[i]->avg_frame_rate);
                }
				if (frame_rate < 1.0 || frame_rate > 65.0)
				{
					frame_rate = av_q2d(av_guess_frame_rate(format_context, streams[i], nullptr));
				}
                break;
            }
            case AVMEDIA_TYPE_AUDIO:
            {
                audio_stream_index = i;
                audio_codecpar = codecpar;
                break;
            }
            default:
                break;
            }
        }
    }

    UrlDemuxer::UrlDemuxer(const std::string& url)
        : Demuxer()
        , _url(url)
    {
    }

    UrlDemuxer::~UrlDemuxer()
    {
    }

    int UrlDemuxer::open(const AVInputFormat** format, AVDictionary** param)
    {
        int res = 0;
        do 
        {
            if ((res = avformat_open_input(&format_context, encodeUrl(_url).c_str(), *format, param)) < 0)
            {
                break;
            }
        } while (false);

        return res;
    }

    std::string UrlDemuxer::encodeUrl(const std::string& url)
    {
        std::string::size_type atPos = url.find_last_of('@');
        // url with password
        if (std::string::npos != atPos)
        {
            std::stringstream ss;

            // found the username:password part
            const char *src = url.c_str();
            std::string::size_type pos = url.find("://");
            if (std::string::npos != pos)
            {
                pos += 3;

                ss << url.substr(0, pos);
            }

            // encoding username:password
            uint8_t ch = 0;
            bool firstColon = true;
            while (pos < atPos)
            {
                ch = src[pos];
                if (ch == ' ')
                {
                    ss << '+';
                }
                else if (ch == ':' && firstColon)
                {
                    ss << ch;
                    firstColon = false;
                }
                else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || strchr("!*()_-.", ch))
                {
                    ss << ch;
                }
                else
                {
                    unsigned char hch = uint8_t((ch >> 4) & 0xf), lch = uint8_t(ch & 0xf);
                    ss << '%' << (unsigned char)(hch > 9 ? hch + 0x37 : hch + 0x30) << (unsigned char)(lch > 9 ? lch + 0x37 : lch + 0x30);
                }
                ++pos;
            }
            ss << url.substr(atPos);

            return ss.str();
        }
        else
        {
            return url;
        }
    }

    UsbDemuxer::UsbDemuxer(const std::string& usb, const std::string& device)
        : Demuxer()
        , _usb(usb), _device(device)
    {
    }

    UsbDemuxer::~UsbDemuxer()
    {
    }

    int UsbDemuxer::open(const AVInputFormat** format, AVDictionary** param)
    {
        int res = 0;
        do 
        {
            std::string usb = "video=" + _usb;
            if ((res = av_dict_set(param, "rtbufsize", "128000000", 0)) < 0)
            {
                break;
            }

            *format = av_find_input_format(_device.c_str());
            if ((res = avformat_open_input(&format_context, usb.c_str(), *format, param)) < 0)
            {
                break;
            }
        } while (false);

        return res;
    }

    DesktopDemuxer::DesktopDemuxer(const std::string& desktop, const std::string& frame_rate, const std::string& device)
        : Demuxer()
        , _desktop(desktop)
        , _frame_rate(frame_rate)
        , _device(device)
    {
    }

    DesktopDemuxer::~DesktopDemuxer()
    {
    }

    int DesktopDemuxer::open(const AVInputFormat** format, AVDictionary** param)
    {
        int res = 0;
        do
        {
            if ((res = av_dict_set(param, "framerate", _frame_rate.c_str(), 0)) < 0)
            {
                break;
            }

            *format = av_find_input_format(_device.c_str());
            if ((res = avformat_open_input(&format_context, _desktop.c_str(), *format, param)) < 0)
            {
                break;
            }
        } while (false);

        return res;
    }

    RawDemuxer::RawDemuxer(int(*raw_data_input)(void*, unsigned char*, int), void* userdata)
        : Demuxer()
        , _raw_data_input(raw_data_input)
        , _userdata(userdata)
        , _io_context(nullptr)
    {
    }

    RawDemuxer::~RawDemuxer()
    {
    }

    int RawDemuxer::open(const AVInputFormat** format, AVDictionary** param)
    {
        static const size_t AVIO_BUFFER_SIZE = 32LL << 20;

        int res = 0;
        do 
        {
            unsigned char* io_buffer = (unsigned char *)av_malloc(AVIO_BUFFER_SIZE);
            if (!io_buffer)
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if (!(_io_context = avio_alloc_context(io_buffer, AVIO_BUFFER_SIZE, 0, _userdata, _raw_data_input, nullptr, nullptr)))
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if ((res = av_probe_input_buffer(_io_context, format, "", nullptr, 0, 0)) < 0)
            {
                break;
            }

            format_context->pb = _io_context;
            if ((res = avformat_open_input(&format_context, nullptr, *format, param)) < 0)
            {
                break;
            }
        } while (false);

        return res;
    }

};


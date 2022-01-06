
#pragma warning (disable:4819)

#include "Muxer.h"

#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavutil/mathematics.h>
#ifdef __cplusplus
}
#endif

#pragma warning (disable:4311)
#pragma warning (disable:4302)

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
        Muxer* muxer = static_cast<Muxer*>(userdata);
        if (muxer->Timeout())
        {
            return 1;
        }
        return 0;
    }

    Muxer::Muxer(const std::string& url, const char* format_name, const AVOutputFormat *oformat)
        : _url(url), _format_name(format_name), _oformat(oformat)
        , format_context(nullptr), streams(nullptr)
        , video_stream_index(-1), video_codecpar(nullptr)
        , audio_stream_index(-1), audio_codecpar(nullptr)
        , _stream_indexs(AVMEDIA_TYPE_NB, -1)
        , _write_timeout(1000LL), _write_at(0LL)
	{
		avformat_alloc_output_context2(&format_context, _oformat, _format_name, _url.c_str());
    }

    Muxer::~Muxer()
    {
    }

    int Muxer::AddStream(const AVCodec* codec, const AVCodecContext* codec_context)
    {
        int res = 0;

        do
        {
            if (!format_context && (res = avformat_alloc_output_context2(&format_context, _oformat, _format_name, _url.c_str())) < 0)
            {
                break;
            }

            AVStream* out_stream = avformat_new_stream(format_context, codec);
            if (!out_stream)
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if ((res = avcodec_parameters_from_context(out_stream->codecpar, codec_context)) < 0)
            {
                break;
            }

            out_stream->time_base = codec_context->time_base;
            out_stream->avg_frame_rate = codec_context->framerate;
            out_stream->r_frame_rate = out_stream->avg_frame_rate;
        } while (false);

        return res;
    }

    int Muxer::AddStream(const AVStream* stream)
    {
        int res = 0;

        do
        {
			if (!format_context && (res = avformat_alloc_output_context2(&format_context, _oformat, _format_name, _url.c_str())) < 0)
			{
				break;
			}

            if (!checkStream(stream))
            {
                break;
            }

            AVStream* out_stream = avformat_new_stream(format_context, nullptr);
            if (!out_stream)
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if ((res = avcodec_parameters_copy(out_stream->codecpar, stream->codecpar)) < 0)
            {
                break;
            }

            out_stream->codecpar->codec_tag = 0;
        } while (false);

        return res;
    }

    int Muxer::AddStream(const AVFormatContext* input_context)
    {
        int res = 0;

        do
        {
			if (!format_context && (res = avformat_alloc_output_context2(&format_context, _oformat, _format_name, _url.c_str())) < 0)
			{
				break;
			}

            for (unsigned int i = 0; i < input_context->nb_streams; ++i)
            {
                const AVStream* stream = input_context->streams[i];

                if (!checkStream(stream))
                {
                    continue;
                }

                AVStream* out_stream = avformat_new_stream(format_context, nullptr);
                if (!out_stream)
                {
					res = AVERROR(ENOMEM);
                    break;
                }

                if ((res = avcodec_parameters_copy(out_stream->codecpar, stream->codecpar)) < 0)
                {
                    break;
                }

                out_stream->codecpar->codec_tag = 0;
            }
        } while (false);

        return res;
    }

    int Muxer::Open(int64_t connect_timeout, int64_t write_timeout, const std::map<std::string, std::string>& options)
    {
        int res = 0;

        AVDictionary* param = nullptr;
        do 
        {
            if (!format_context)
            {
                break;
            }

            //av_dump_format(format_context, 0, url.c_str(), 1);

            // set connection timeout
            _write_timeout = connect_timeout;
            _write_at = milliseconds_since_epoch();
            format_context->interrupt_callback = AVIOInterruptCB{ InterruptCb, this };

            // unless it's a no file (we'll talk later about that) write to the disk (FLAG_WRITE)
            // but basically it's a way to save the file to a buffer so you can store it
            // wherever you want.
            if (!(format_context->oformat->flags & AVFMT_NOFILE)) 
            {
                if ((res = avio_open(&format_context->pb, _url.c_str(), AVIO_FLAG_WRITE)) < 0) 
                {
                    break;
                }
            }

            // update write timeout
            _write_timeout = write_timeout;

            for (std::map<std::string, std::string>::const_iterator optr = options.cbegin(); optr != options.cend() && res >= 0; ++optr)
            {
                res = av_dict_set(&param, optr->first.c_str(), optr->second.c_str(), 0);
            }
            if (res < 0)
            {
                break;
            }

            if ((res = avformat_write_header(format_context, &param)) < 0)
            {
                avio_closep(&format_context->pb);

                avformat_free_context(format_context);
                format_context = nullptr;
                break;
            }

            parseParameters();
        } while (false);

        if (param)
        {
            av_dict_free(&param);
        }

        return res;
    }

    int Muxer::Write(const AVPacket* packet)
    {
        int res = 0;
        if (_stream_indexs[(int)(packet->opaque)] >= 0)
        {
            AVPacket tomux;
            if ((res = av_packet_ref(&tomux, packet)) >= 0)
            {
                tomux.stream_index = _stream_indexs[(int)(packet->opaque)];

                const AVRational& in_time_base = packet->time_base;
                const AVRational& out_time_base = format_context->streams[tomux.stream_index]->time_base;
                /* copy packet */
                tomux.pts = av_rescale_q_rnd(tomux.pts, in_time_base, out_time_base, static_cast<AVRounding>(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                tomux.dts = av_rescale_q_rnd(tomux.dts, in_time_base, out_time_base, static_cast<AVRounding>(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                tomux.duration = av_rescale_q(tomux.duration, in_time_base, out_time_base);
                tomux.pos = -1;

                _write_at = milliseconds_since_epoch();
                res = av_interleaved_write_frame(format_context, &tomux);

                // un-reference to free
                av_packet_unref(&tomux);
            }
        }

        return res;
    }

    bool Muxer::Timeout()
    {
        return _write_at + _write_timeout <= milliseconds_since_epoch();
    }

    int Muxer::Close()
    {
        int res = 0;
        if (format_context)
        {
            if (format_context->pb)
            {
                _write_at = milliseconds_since_epoch();

                res = av_write_trailer(format_context);

                // ignore the result, just to release the resource
                avio_closep(&format_context->pb);
            }

            avformat_free_context(format_context);
            format_context = nullptr;
        }

        _stream_indexs.clear();

        return res;
    }

    bool Muxer::checkStream(const AVStream* stream)
    {
        switch (stream->codecpar->codec_type)
        {
        case AVMEDIA_TYPE_VIDEO:
        case AVMEDIA_TYPE_AUDIO:
        case AVMEDIA_TYPE_SUBTITLE:
            return true;
        case AVMEDIA_TYPE_DATA:
        case AVMEDIA_TYPE_ATTACHMENT:
        case AVMEDIA_TYPE_NB:
        default:
            return false;
        }
    }

    void Muxer::parseParameters()
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

            _stream_indexs[codecpar->codec_type] = i;
        }
    }

};


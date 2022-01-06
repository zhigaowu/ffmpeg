// ffmpeg.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#pragma warning (disable:4819)
#pragma warning (disable:4996)

#include "Demuxer.h"
#include "Decoder.h"

#include "MatConverter.h"
#include "GpuMatConverter.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>

#ifdef __cplusplus
}
#endif

#include <iostream>


static int64_t microseconds_since_epoch()
{
    typedef std::chrono::microseconds time_precision;
    typedef std::chrono::time_point<std::chrono::system_clock, time_precision> time_point_precision;
    time_point_precision tp = std::chrono::time_point_cast<time_precision>(std::chrono::system_clock::now());
    return tp.time_since_epoch().count();
}

static int DecodeH264Initialize(AVCodecContext* codec_context, void* userdata)
{
    const AVCodecHWConfig* device_config = (const AVCodecHWConfig*)codec_context->opaque;

    if (device_config)
	{
		if ((device_config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) == AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX)
		{
			codec_context->get_format = ffmpeg::frames_decode_context_initialize;
		}
		else if ((device_config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) == AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)
		{
			codec_context->get_format = ffmpeg::device_decode_context_initialize;
		}
    }

    return 0;
}

static cv::cuda::GpuMat image;
static cv::Mat show;

int test_decode(int argc, char** argv)
{
    int res = 0;
    char error[128] = { 0 };
    
    int device_index = -1;
    enum AVHWDeviceType device_type = av_hwdevice_find_type_by_name(argv[2]);
    if (device_type == AV_HWDEVICE_TYPE_CUDA)
    {
        device_index = atoi(argv[2]);
        cv::cuda::setDevice(device_index);
    }

    enum AVMediaType media_type = AVMEDIA_TYPE_VIDEO;

    ffmpeg::Demuxer* demuxer = nullptr; // ffmpeg.exe[0] case_name[1] device_name[2] device_index[3] url_type[4] src_url[5] dst_url[6]
    if (strcmp(argv[4], "url") == 0)
    {
        demuxer = new ffmpeg::UrlDemuxer(argv[5]);
    }
    else if (strcmp(argv[4], "usb") == 0)
    {
        demuxer = new ffmpeg::UsbDemuxer(argv[5], "dshow");
    }
    else if (strcmp(argv[4], "desktop") == 0)
    {
        demuxer = new ffmpeg::DesktopDemuxer("desktop", "25", "gdigrab");
    }
    else
    {
        demuxer = new ffmpeg::UrlDemuxer(argv[5]);
    }

    do 
    {
        {
            std::map<std::string, std::string> options;
            options.insert(std::make_pair("rtsp_transport", "tcp"));

            if ((res = demuxer->Open(5000LL, 500LL, options, 30LL * AV_TIME_BASE)) < 0)
            {
                break;
            }

            AVStream* vs = demuxer->streams[demuxer->video_stream_index];
            std::cout << "fps: " << demuxer->frame_rate << std::endl;
            std::cout << "timebase: " << vs->time_base.num << "/" << vs->time_base.den << std::endl;
        }

        cv::Size target_size(demuxer->video_codecpar->width, demuxer->video_codecpar->height);

        ffmpeg::Decoder decoder(media_type);
        
        std::map<std::string, std::string> decode_options;
        {
            //decode_options.insert(std::make_pair("strict", "1"));
        }

		std::map<std::string, std::string> device_options;
		{
			device_options.insert(std::make_pair("child_device_type", "dxva2"));
			device_options.insert(std::make_pair("child_device", "dxva2"));
		}
        
        if ((res = decoder.Create(demuxer->format_context, device_type, device_index, DecodeH264Initialize, AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX, nullptr, decode_options, device_options)) < 0)
        {
            break;
        }

        if (device_type == AV_HWDEVICE_TYPE_CUDA)
        {
            cv::cuda::Stream stream;
            ffmpeg::GpuMatConverter converter(device_type, device_index, target_size);

            AVPacket* packet = av_packet_alloc();
            while ((res = demuxer->Read(&packet) >= 0))
            {
                if (media_type == (int)packet->opaque)
                {
                    AVRational tb = demuxer->streams[demuxer->video_stream_index]->time_base;
                    std::cout << "pts:" << packet->pts << " timestamp:" << 1.0 * packet->pts * tb.num / tb.den << std::endl;

                    if ((res = decoder.Decode(
                        packet, 
                        [&converter, &stream](AVFrame* frame) {
                        int64_t tss = microseconds_since_epoch();
                        int res = converter.Convert(frame, image, stream);
                        int64_t tse = microseconds_since_epoch();

                        std::cout << "convert costs: " << tse - tss << "us" << std::endl;

                        if (res >= 0)
                        {
                            image.download(show);
                            cv::imshow("test", show);
                            cv::waitKey(10);
                        }
                        else
                        {
                            std::cout << "convert image failed: " << res << std::endl;
                        }
                    },
                    demuxer->seek_pts)) < 0)
                    {
                        break;
                    }
                }

                av_packet_free(&packet);

                packet = av_packet_alloc();
            }
            av_packet_free(&packet);
        } 
        else
        {
            ffmpeg::MatConverter converter(device_type, target_size);

            AVPacket* packet = av_packet_alloc();
            while ((res = demuxer->Read(&packet) >= 0))
            {
                if (packet->stream_index == demuxer->video_stream_index)
                {
                    AVRational tb = demuxer->streams[demuxer->video_stream_index]->time_base;
                    std::cout << "pts:" << packet->pts << " timestamp:" << 1.0 * packet->pts * tb.num / tb.den << std::endl;
                    
                    if ((res = decoder.Decode(
                        packet,
                        [&converter](AVFrame* frame) {
                        int64_t tss = microseconds_since_epoch();
                        int res = converter.Convert(frame, show);
                        int64_t tse = microseconds_since_epoch();

                        std::cout << "convert costs: " << tse - tss << "us" << std::endl;

                        if (res >= 0)
                        {
                            cv::imshow("test", show);
                            cv::waitKey(10);
                        }
                        else
                        {
                            std::cout << "convert image failed: " << res << std::endl;
                        }
                    },
                        demuxer->seek_pts)) < 0)
                    {
                        break;
                    }
                }

                av_packet_free(&packet);

                packet = av_packet_alloc();
            }
            av_packet_free(&packet);
        }

        decoder.Destroy();
    } while (false);

    if (res < 0)
    {
        char error[256] = { 0 };
        av_strerror(res, error, sizeof(error));
        std::cout << error << std::endl;

        std::cin >> error;
    }

    if (demuxer)
    {
        demuxer->Close();
        delete demuxer;
    }

    return res;
}

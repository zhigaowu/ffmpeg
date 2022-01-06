// ffmpeg.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#pragma warning (disable:4819)
#pragma warning (disable:4996)

#include "Demuxer.h"
#include "Muxer.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>

#ifdef __cplusplus
}
#endif

#include <iostream>


int test_transmux(int argc, char** argv)
{
    int res = 0;
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

    ffmpeg::Muxer muxer(std::string(argv[6]) + "transmux.mp4");
    do 
    {
        {
            std::map<std::string, std::string> demux_options;
            demux_options.insert(std::make_pair("rtsp_transport", "tcp"));
            if ((res = demuxer->Open(5000LL, 500LL, demux_options)) < 0)
            {
                break;
            }

            AVStream* vs = demuxer->format_context->streams[demuxer->video_stream_index];
            std::cout << "fps: " << vs->avg_frame_rate.num << "/" << vs->avg_frame_rate.den << std::endl;
            std::cout << "timebase: " << vs->time_base.num << "/" << vs->time_base.den << std::endl;
        }

        {
            char value[128] = { 0 };
            if ((res = muxer.AddStream(demuxer->format_context)) < 0)
            {
                break;
            }

            std::map<std::string, std::string> mux_options;
            if ((res = muxer.Open(5000LL, 500LL, mux_options)) < 0)
            {
                break;
            }
        }

        AVPacket* packet = av_packet_alloc();
        while ((res = demuxer->Read(&packet)) >= 0)
        {
            if ((res = muxer.Write(packet)) < 0)
            {
                break;
            }

            av_packet_free(&packet);

            packet = av_packet_alloc();
        }
        av_packet_free(&packet);
    } while (false);

    muxer.Close();

    if (demuxer)
    {
        demuxer->Close();
        delete demuxer;
    }

    if (res < 0)
    {
        char error[256] = { 0 };
        av_strerror(res, error, sizeof(error));
        std::cout << error << std::endl;

        std::cin >> error;
    }

    return res;
}

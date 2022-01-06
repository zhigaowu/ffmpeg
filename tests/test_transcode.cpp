// ffmpeg.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#pragma warning (disable:4819)
#pragma warning (disable:4996)

#include "Demuxer.h"
#include "Decoder.h"

#include "PixelFmtConverter.h"

#include "Muxer.h"
#include "Encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>

#ifdef __cplusplus
}
#endif

#include <thread>
#include <chrono>
#include <iostream>

static int EncodeH264Initialize(AVCodecContext* codec_context, void* userdata)
{
    AVCodecContext* dec_context = (AVCodecContext*)userdata;

    codec_context->height = dec_context->height;
    codec_context->width = dec_context->width;
    codec_context->pix_fmt = AV_PIX_FMT_NV12;
    codec_context->sample_aspect_ratio = AVRational{1,1};

    codec_context->bit_rate = /*1024 * 1024;// */dec_context->bit_rate;
    codec_context->gop_size = 10;

    /* video time_base can be set to whatever is handy and supported by encoder */
    codec_context->framerate = dec_context->framerate;
    codec_context->time_base = av_inv_q(dec_context->framerate);

    return ffmpeg::frames_encode_context_initialize(codec_context, codec_context->pix_fmt);
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

int test_transcode(int argc, char** argv)
{
    int res = 0;
    
    int device_index = -1;
    enum AVHWDeviceType device_type = av_hwdevice_find_type_by_name(argv[2]);
    if (device_type == AV_HWDEVICE_TYPE_CUDA)
    {
        device_index = atoi(argv[2]);
    }

    enum AVMediaType media_type = AVMEDIA_TYPE_VIDEO;

    ffmpeg::PixelFmtConverter* converter = nullptr;
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
        converter = new ffmpeg::PixelFmtConverter(device_type, AV_PIX_FMT_YUV420P);
    }
    else
    {
        demuxer = new ffmpeg::UrlDemuxer(argv[5]);
    }

    ffmpeg::Muxer muxer(std::string(argv[6]) + "transcode.mp4", argv[7]);
    do 
    {
        ffmpeg::Decoder decoder(media_type);
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

            std::map<std::string, std::string> decode_options;
            decode_options.insert(std::make_pair("tune", "zerolatency"));

			std::map<std::string, std::string> device_options;
			device_options.insert(std::make_pair("child_device_type", "d3d11va"));
			device_options.insert(std::make_pair("child_device", "d3d11va"));
            if ((res = decoder.Create(demuxer->format_context, device_type /*AV_HWDEVICE_TYPE_NONE*/, device_index, DecodeH264Initialize, AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX, nullptr, decode_options, device_options)) < 0)
            {
                break;
            }
        }

        /*
          preset: ultrafast superfast veryfast faster fast medium slow slower veryslow placebo
            tune: film animation grain stillimage psnr ssim fastdecode zerolatency
        */

        ffmpeg::Encoder encoder(media_type);
        {
            char value[128] = { 0 };
            std::map<std::string, std::string> encode_options;
            if (AV_HWDEVICE_TYPE_CUDA == device_type)
            {
                encode_options.insert(std::make_pair("tune", "ull"));
            } 
            else
            {
                encode_options.insert(std::make_pair("tune", "zerolatency"));
            }

			std::map<std::string, std::string> device_options;
			device_options.insert(std::make_pair("child_device_type", "dxva2"));
			device_options.insert(std::make_pair("child_device", "dxva2"));
            if ((res = encoder.Create(AV_CODEC_ID_H264, muxer.format_context->oformat->flags, device_type, device_index, EncodeH264Initialize, 2, decoder.decode_context->codec_context, encode_options, device_options)) < 0)
            {
                break;
            }

            if ((res = muxer.AddStream(encoder.encode_context->codec, encoder.encode_context->codec_context)) < 0)
            {
                break;
            }

            std::map<std::string, std::string> mux_options;
			mux_options.insert(std::make_pair("rtsp_transport", "tcp"));
            if ((res = muxer.Open(5000LL, 500LL, mux_options)) < 0)
            {
                break;
            }
        }

        AVRational input_timebase = demuxer->streams[demuxer->video_stream_index]->time_base;

        int frames_encoded = 0;

        do 
        {
			AVPacket* packet = av_packet_alloc();
			while ((res = demuxer->Read(&packet)) >= 0)
			{
				if (media_type == (int)packet->opaque)
				{
					if (converter)
					{
						packet->pts = av_rescale_q(frames_encoded, decoder.decode_context->codec_context->time_base, packet->time_base);
						packet->dts = packet->pts;
						packet->duration = 0;
						frames_encoded++;
					}

					std::this_thread::sleep_for(std::chrono::milliseconds(30));

					std::cout << "pts:" << packet->pts << " timestamp:" << 1.0 * packet->pts * packet->time_base.num / packet->time_base.den << std::endl;
					if ((res = decoder.Decode(packet, [&encoder, &muxer, converter, input_timebase](AVFrame* frame) {
						int res = 0;
						do
						{
							AVFrame* dst_frame = frame;
							if (converter && (res = converter->Convert(frame, &dst_frame)) < 0)
							{
								break;
							}
							res = encoder.Encode(dst_frame, input_timebase, [&muxer](AVPacket* pkt) {
								int res = muxer.Write(pkt);
								if (res < 0)
								{
									char error[256] = { 0 };
									av_strerror(res, error, sizeof(error));
									std::cout << "mux failed: " << error << std::endl;
								}
							});
						} while (false);
						if (res < 0)
						{
							char error[256] = { 0 };
							av_strerror(res, error, sizeof(error));
							std::cout << "encode failed: " << error << std::endl;
						}
					}, demuxer->seek_pts)) < 0)
					{
						break;
					}
				}

				av_packet_free(&packet);

				packet = av_packet_alloc();
			}
			av_packet_free(&packet);

        } while (false);

        encoder.Destroy();
        decoder.Destroy();
    } while (false);

    muxer.Close();
    
    if (demuxer)
    {
        demuxer->Close();
        delete demuxer;
    }

    char error[256] = { 0 };
    if (res < 0)
    {
        av_strerror(res, error, sizeof(error));
        std::cout << error << std::endl;
    }

    std::cin >> error;

    return res;
}

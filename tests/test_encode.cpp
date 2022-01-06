#pragma warning (disable:4819)
#pragma warning (disable:4996)

#include "Muxer.h"
#include "Encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>

#include "libavutil/imgutils.h"

#ifdef __cplusplus
}
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <nppdefs.h>
#include <nppi_color_conversion.h>

#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>

struct EncodeInfo 
{
	int width = 0;
	int height = 0;
	AVPixelFormat format = AV_PIX_FMT_YUV420P;
};

static int EncodeH264Initialize(AVCodecContext* codec_context, void* userdata)
{
	EncodeInfo* info = (EncodeInfo*)userdata;

	codec_context->height = info->height;
	codec_context->width = info->width;
	codec_context->pix_fmt = info->format;
	codec_context->sample_aspect_ratio = AVRational{ 1,1 };

	codec_context->bit_rate = 1024 * 1024 * 4;
	codec_context->gop_size = 10;

	/* video time_base can be set to whatever is handy and supported by encoder */
	codec_context->framerate = av_make_q(25, 1);
	codec_context->time_base = av_inv_q(codec_context->framerate);

	return ffmpeg::frames_encode_context_initialize(codec_context, codec_context->pix_fmt);
}

int test_encode(int argc, char** argv)
{
	int res = 0;

	EncodeInfo info;
	int device_index = -1;
	enum AVHWDeviceType device_type = av_hwdevice_find_type_by_name(argv[2]);
	if (device_type == AV_HWDEVICE_TYPE_CUDA)
	{
		device_index = atoi(argv[2]);
		cv::cuda::setDevice(device_index);
	}

	enum AVMediaType media_type = AVMEDIA_TYPE_VIDEO;

	// ffmpeg.exe[0] case_name[1] device_name[2] device_index[3] url_type[4] src_url[5] dst_url[6]

	std::vector<std::string> images;
	cv::glob(argv[5] + std::string("*.jpg"), images);

	std::sort(images.begin(), images.end());

	cv::Mat image = cv::imread(images[0]);
	info.width = image.cols;
	info.height = image.rows;

	AVFrame* frame = av_frame_alloc();
	ffmpeg::Muxer muxer(std::string(argv[6]) + "encode.mp4", argv[7]);
	do
	{
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
			if ((res = encoder.Create(AV_CODEC_ID_H264, muxer.format_context->oformat->flags, device_type, device_index, EncodeH264Initialize, 2, &info, encode_options, device_options)) < 0)
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

		AVRational input_timebase = av_make_q(1, AV_TIME_BASE);
		int frames_encoded = 0;

		frame->width = info.width;
		frame->height = info.height;
		frame->format = info.format;
		frame->time_base = input_timebase;

		if (device_index < 0)
		{
			cv::Mat cvt;
			for (size_t i = 0; i < images.size(); ++i)
			{
				cv::cvtColor(cv::imread(images[i]), cvt, cv::COLOR_BGR2YUV_I420);
				if ((res = av_image_fill_arrays(frame->data, frame->linesize, cvt.data, info.format, info.width, info.height, 1)) < 0)
				{
					break;
				}

				frame->pts = AV_TIME_BASE * (frames_encoded++) / 25 ;
				frame->pkt_dts = frame->pts;
				frame->pkt_duration = 0;

				std::cout << "pts:" << frame->pts << " timestamp:" << 1.0 * frame->pts * frame->time_base.num / frame->time_base.den << std::endl;

				res = encoder.Encode(frame, input_timebase, [&muxer](AVPacket* pkt) {
					int res = muxer.Write(pkt);
					if (res < 0)
					{
						char error[256] = { 0 };
						av_strerror(res, error, sizeof(error));
						std::cout << "mux failed: " << error << std::endl;
					}
				});
			}
		} 
		else
		{
			if ((res = av_hwframe_get_buffer(encoder.encode_context->codec_context->hw_frames_ctx, frame, 0)) < 0)
			{
				break;
			}

			cv::cuda::Stream stream;
			cv::cuda::GpuMat bgr;
			NppiSize npp_size;
			npp_size.width = info.width;
			npp_size.height = info.height;
			for (size_t i = 0; i < images.size(); ++i)
			{
				bgr.upload(cv::imread(images[i]), stream);
				stream.waitForCompletion();

				NppStatus nppres = nppiBGRToYCbCr420_8u_C3P3R(bgr.data, bgr.step, frame->data, frame->linesize, npp_size);

				frame->pts = AV_TIME_BASE * (frames_encoded++) / 25;
				frame->pkt_dts = frame->pts;
				frame->pkt_duration = 0;

				std::cout << "pts:" << frame->pts << " timestamp:" << 1.0 * frame->pts * frame->time_base.num / frame->time_base.den << std::endl;

				res = encoder.Encode(frame, input_timebase, [&muxer](AVPacket* pkt) {
					int res = muxer.Write(pkt);
					if (res < 0)
					{
						char error[256] = { 0 };
						av_strerror(res, error, sizeof(error));
						std::cout << "mux failed: " << error << std::endl;
					}
				});
			}
		}

		encoder.Destroy();
	} while (false);

	av_frame_free(&frame);

	muxer.Close();

	char error[256] = { 0 };
	if (res < 0)
	{
		av_strerror(res, error, sizeof(error));
		std::cout << error << std::endl;
	}

	std::cin >> error;

	return res;
}


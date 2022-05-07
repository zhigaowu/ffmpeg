
#pragma warning (disable:4819)

#include "MatConverter.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/imgutils.h>

#ifdef __cplusplus
}
#endif

#include <sstream>
#include <chrono>

namespace ffmpeg 
{
    MatConverter::MatConverter(enum AVHWDeviceType device_type, const cv::Size& target_size)
        : _device_type(device_type)
        , _origin_size(0, 0), _target_size(target_size)
        , _image_mapper(av_frame_alloc()), _device_mapper(av_frame_alloc())
        , _sws_context(nullptr)
    {
    }

    MatConverter::~MatConverter()
    {
        av_frame_free(&_image_mapper);
        av_frame_free(&_device_mapper);

        if (_sws_context)
        {
            sws_freeContext(_sws_context);
            _sws_context = nullptr;
        }
    }

    int MatConverter::Convert(AVFrame* frame, cv::Mat& image)
    {
        int res = 0;
        do 
        {
            AVFrame* real_frame = frame;
            if (_device_type != AV_HWDEVICE_TYPE_NONE)
            {
                if ((res = av_hwframe_transfer_data(_device_mapper, frame, 0)) < 0)
                {
                    break;
                }
                real_frame = _device_mapper;
            }

			if (!_target_size.empty())
			{
				if (!_sws_context)
				{
					_origin_size = cv::Size(frame->width, frame->height);
					_sws_context = sws_getContext(frame->width, frame->height, (AVPixelFormat)real_frame->format, _target_size.width, _target_size.height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
				}
				else
				{
					if (_origin_size.width != frame->width || _origin_size.height != frame->height)
					{
						_origin_size = cv::Size(frame->width, frame->height);

						sws_freeContext(_sws_context);
						_sws_context = sws_getContext(frame->width, frame->height, (AVPixelFormat)real_frame->format, _target_size.width, _target_size.height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
					}
				}

				// check image size and adjust it
				if (image.size() != _target_size)
				{
					image = cv::Mat(_target_size, CV_8UC3);
				}
			} 
			else
			{
				if (!_sws_context)
				{
					_origin_size = cv::Size(frame->width, frame->height);
					_sws_context = sws_getContext(frame->width, frame->height, (AVPixelFormat)real_frame->format, _origin_size.width, _origin_size.height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
				}
				else
				{
					if (_origin_size.width != frame->width || _origin_size.height != frame->height)
					{
						_origin_size = cv::Size(frame->width, frame->height);

						sws_freeContext(_sws_context);
						_sws_context = sws_getContext(frame->width, frame->height, (AVPixelFormat)real_frame->format, _origin_size.width, _origin_size.height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
					}
				}

				// check image size and adjust it
				if (image.size() != _origin_size)
				{
					image = cv::Mat(_origin_size, CV_8UC3);
				}
			}

            // map resource
            if ((res = av_image_fill_arrays(_image_mapper->data, _image_mapper->linesize, image.data, AV_PIX_FMT_BGR24, image.cols, image.rows, 1)) < 0)
            {
                break;
            }

            // convert and scale
            res = sws_scale(_sws_context, (const unsigned char* const*)real_frame->data, real_frame->linesize, 0, frame->height, _image_mapper->data, _image_mapper->linesize);
        } while (false);

        return res;
    }

};



#pragma warning (disable:4819)

#include "PixelFmtConverter.h"

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
    PixelFmtConverter::PixelFmtConverter(enum AVHWDeviceType device_type, enum AVPixelFormat pixel_format, int width, int height)
        : _device_type(device_type)
        , _pixel_format(pixel_format)
        , _width(width), _height(height)
        , _image_mapper(av_frame_alloc()), _device_mapper(av_frame_alloc())
        , _sws_context(nullptr)
    {
    }

    PixelFmtConverter::~PixelFmtConverter()
    {
        av_frame_free(&_image_mapper);
        av_frame_free(&_device_mapper);

        if (_sws_context)
        {
            sws_freeContext(_sws_context);
            _sws_context = nullptr;
        }
    }

    int PixelFmtConverter::Convert(AVFrame* src_frame, AVFrame** dst_frame)
    {
        int res = 0;
        do 
        {
            AVFrame* real_frame = src_frame;
            if (_device_type != AV_HWDEVICE_TYPE_NONE)
            {
                if ((res = av_hwframe_transfer_data(_device_mapper, src_frame, 0)) < 0)
                {
                    break;
                }
                real_frame = _device_mapper;
            }

            if (!_sws_context)
            {
                if (_width <= 0)
                {
                    _width = src_frame->width;
                }

                if (_height <= 0)
                {
                    _height = src_frame->height;
                }

                _sws_context = sws_getContext(src_frame->width, src_frame->height, (AVPixelFormat)real_frame->format, _width, _height, _pixel_format, SWS_BICUBIC, NULL, NULL, NULL);

                _image_mapper->width = _width;
                _image_mapper->height = _height;
                _image_mapper->format = _pixel_format;
                if ((res = av_frame_get_buffer(_image_mapper, 1)) < 0)
                {
                    break;
                }
            }

            // convert and scale
            if ((res = sws_scale(_sws_context, (const unsigned char* const*)real_frame->data, real_frame->linesize, 0, src_frame->height, _image_mapper->data, _image_mapper->linesize)) < 0)
            {
                *dst_frame = nullptr;
            }
            else
            {
                _image_mapper->pts = src_frame->pts;
                _image_mapper->pkt_dts = src_frame->pkt_dts;
                _image_mapper->pkt_duration = src_frame->pkt_duration;

                _image_mapper->time_base = src_frame->time_base;
                _image_mapper->opaque = src_frame->opaque;

                *dst_frame = _image_mapper;
            }
        } while (false);

        return res;
    }

};


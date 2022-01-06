
/*   Copyright [2022] [wuzhigaoem@gmail.com]
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _FFMPEG_PIXEL_FORMAT_CONVERTER_HEADER_H_
#define _FFMPEG_PIXEL_FORMAT_CONVERTER_HEADER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <libswscale/swscale.h>

#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>

#ifdef __cplusplus
}
#endif

namespace ffmpeg
{
    class PixelFmtConverter
    {
    public:
        explicit PixelFmtConverter(enum AVHWDeviceType device_type, enum AVPixelFormat pixel_format, int width = 0, int height = 0);
        ~PixelFmtConverter();

        int Convert(AVFrame* src_frame, AVFrame** dst_frame);

    private:
        enum AVHWDeviceType _device_type;
        enum AVPixelFormat _pixel_format;

    private:
        int _width;
        int _height;

    private:
        AVFrame* _image_mapper;
        AVFrame* _device_mapper;

    private:
        SwsContext* _sws_context;

    private:
        PixelFmtConverter() = delete;

        PixelFmtConverter(PixelFmtConverter& rhs) = delete;
        PixelFmtConverter& operator=(PixelFmtConverter& rhs) = delete;

        PixelFmtConverter(PixelFmtConverter&& rhs) = delete;
        PixelFmtConverter& operator=(PixelFmtConverter&& rhs) = delete;
    };
};

#endif



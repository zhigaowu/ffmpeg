
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

#ifndef _FFMPEG_GPUMAT_CONVERTER_HEADER_H_
#define _FFMPEG_GPUMAT_CONVERTER_HEADER_H_

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>

#ifdef __cplusplus
}
#endif

#include <nppdefs.h>

namespace ffmpeg
{
    class GpuMatConverter
    {
    public:
        explicit GpuMatConverter(enum AVHWDeviceType device_type, int device_index, const cv::Size& target_size);
        ~GpuMatConverter();

        int Convert(AVFrame* frame, cv::cuda::GpuMat& image, cv::cuda::Stream& stream);

    private:
		enum AVHWDeviceType _device_type;

    private:
        cv::Size _target_size;

    private:
        cv::cuda::GpuMat _temp;

	private:
		NppStreamContext _npp_stream_ctx;
		NppiSize _npp_size;

    private:
        GpuMatConverter() = delete;

        GpuMatConverter(GpuMatConverter& rhs) = delete;
        GpuMatConverter& operator=(GpuMatConverter& rhs) = delete;

        GpuMatConverter(GpuMatConverter&& rhs) = delete;
        GpuMatConverter& operator=(GpuMatConverter&& rhs) = delete;
    };
};

#endif











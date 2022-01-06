
#pragma warning (disable:4819)

#include "GpuMatConverter.h"

#include <opencv2/cudawarping.hpp>

#include <cuda_runtime_api.h>

#include <nppi_color_conversion.h>

#include <sstream>

#include <chrono>

#define CUDA_STREAM(stream) static_cast<cudaStream_t>(stream.cudaPtr())

namespace ffmpeg 
{
    GpuMatConverter::GpuMatConverter(enum AVHWDeviceType device_type, int device_index, const cv::Size& target_size)
        : _device_type(device_type)
        , _target_size(target_size)
        , _temp()
		, _npp_stream_ctx(), _npp_size()
    {
		do 
		{
			_npp_stream_ctx.nCudaDeviceId = device_index;

			cudaError_t res = cudaDeviceGetAttribute(&_npp_stream_ctx.nCudaDevAttrComputeCapabilityMajor,
				cudaDevAttrComputeCapabilityMajor,
				_npp_stream_ctx.nCudaDeviceId);
			if (res != cudaSuccess)
			{
				break;
			}

			res = cudaDeviceGetAttribute(&_npp_stream_ctx.nCudaDevAttrComputeCapabilityMinor,
				cudaDevAttrComputeCapabilityMinor,
				_npp_stream_ctx.nCudaDeviceId);
			if (res != cudaSuccess)
			{
				break;
			}

			res = cudaStreamGetFlags(_npp_stream_ctx.hStream, &_npp_stream_ctx.nStreamFlags);
			if (res != cudaSuccess)
			{
				break;
			}

			cudaDeviceProp oDeviceProperties;
			res = cudaGetDeviceProperties(&oDeviceProperties, _npp_stream_ctx.nCudaDeviceId);
			if (res != cudaSuccess)
			{
				break;
			}

			_npp_stream_ctx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
			_npp_stream_ctx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
			_npp_stream_ctx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
			_npp_stream_ctx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
		} while (false);
    }

    GpuMatConverter::~GpuMatConverter()
    {
    }

    int GpuMatConverter::Convert(AVFrame* frame, cv::cuda::GpuMat& image, cv::cuda::Stream& stream)
    {
		NppStatus res = NPP_SUCCESS;
        do 
        {
            if (_device_type != AV_HWDEVICE_TYPE_CUDA)
            {
                res = NPP_BAD_ARGUMENT_ERROR;
                break;
            }

			_npp_stream_ctx.hStream = CUDA_STREAM(stream);

			if (image.size() != _target_size)
			{
				image = cv::cuda::GpuMat(_target_size, CV_8UC3);
			}

			if (_npp_size.width != frame->height || _npp_size.height != frame->width)
			{
				_npp_size.width = frame->width;
				_npp_size.height = frame->height;
			}

			if (_npp_size.width == image.cols && _npp_size.height == image.rows)
			{
				_temp = image;
			} 
			else
			{
				if (_npp_size.width != _temp.cols || _npp_size.height != _temp.rows)
				{
					_temp = cv::cuda::GpuMat(_npp_size.height, _npp_size.width, CV_8UC3);
				}
			}

			switch (frame->format)
			{
			case AV_PIX_FMT_CUDA:
			case AV_PIX_FMT_NV12:
			{
				res = nppiNV12ToBGR_8u_P2C3R_Ctx(frame->data, frame->linesize[0], _temp.data, (int)_temp.step, _npp_size, _npp_stream_ctx);
				break;
			}
			default:
				res = NPP_DATA_TYPE_ERROR;
			}

            if (_temp.size() != image.size())
			{
				cv::cuda::resize(_temp, image, _target_size, 0.0, 0.0, cv::INTER_LINEAR, stream);
            }

            if (NPP_SUCCESS == res)
            {
                stream.waitForCompletion();
            }
        } while (false);

        return res;
    }

};


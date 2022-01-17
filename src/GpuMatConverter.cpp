
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
		, _npp_stream_ctx()
		, _target_size(target_size)
		, _buffer()
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

			NppiSize npp_size{ frame->width, frame->height };
			if (!_target_size.empty())
			{
				if (image.empty() || image.cols != _target_size.width || image.rows != _target_size.height)
				{
					image = cv::cuda::GpuMat(_target_size.height, _target_size.width, CV_8UC3);
				}

				if (npp_size.width == _target_size.width && npp_size.height == _target_size.height)
				{
					_buffer = image;
				} 
				else
				{
					if (_buffer.empty() || _buffer.cols != npp_size.width || _buffer.rows != npp_size.height)
					{
						_buffer = cv::cuda::GpuMat(npp_size.height, npp_size.width, CV_8UC3);
					}
				}
			}
			else
			{
				if (image.empty() || image.cols != npp_size.width || image.rows != npp_size.height)
				{
					image = cv::cuda::GpuMat(npp_size.height, npp_size.width, CV_8UC3);
				}
				_buffer = image;
			}

			switch (frame->format)
			{
			case AV_PIX_FMT_CUDA:
			case AV_PIX_FMT_NV12:
			{
				res = nppiNV12ToBGR_8u_P2C3R_Ctx(frame->data, frame->linesize[0], _buffer.data, (int)_buffer.step, npp_size, _npp_stream_ctx);
				break;
			}
			default:
				res = NPP_DATA_TYPE_ERROR;
			}

			if (_buffer.size() != image.size())
			{
				cv::cuda::resize(_buffer, image, _target_size, 0, 0, cv::INTER_LINEAR, stream);
			}

            if (NPP_SUCCESS == res)
            {
                stream.waitForCompletion();
            }
        } while (false);

        return res;
    }

};


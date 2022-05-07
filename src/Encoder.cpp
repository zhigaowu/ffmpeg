
#pragma warning (disable:4819)

#include "Encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext_qsv.h>

#ifdef __cplusplus
}
#endif

#include <vector>
#include <functional>

namespace ffmpeg
{
    int frames_encode_context_initialize(AVCodecContext* ctx, enum AVPixelFormat format)
    {
        int res = 0;

        do
        {
            const AVCodecHWConfig* device_config = static_cast<const AVCodecHWConfig*>(ctx->opaque);

            if (device_config)
            {
                /* create a pool of surfaces to be used by the decoder */
                if (!(ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->hw_device_ctx)))
                {
					res = AVERROR(ENOMEM);
                    break;
                }

                AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(ctx->hw_frames_ctx->data);

                frames_ctx->format = device_config->pix_fmt;
                frames_ctx->sw_format = format;
                frames_ctx->width = FFALIGN(ctx->width, 32);
                frames_ctx->height = FFALIGN(ctx->height, 32);
                frames_ctx->initial_pool_size = 10;

                if ((res = av_hwframe_ctx_init(ctx->hw_frames_ctx)) < 0)
                {
                    av_buffer_unref(&(ctx->hw_frames_ctx));
                    break;
                }

                // encode context pixel format should be the same as the hardware frames format
                ctx->pix_fmt = frames_ctx->format;
            }
        } while (false);

        return res;
    }

    CodContextInitializer::CodContextInitializer()
        : codec(nullptr), codec_context(nullptr)
    {
    }

    CodContextInitializer::~CodContextInitializer()
    {
    }

    int CodContextInitializer::Initialize(enum AVCodecID, int, enum AVHWDeviceType, int, int, const std::map<std::string, std::string>&)
    {
        return AVERROR_PATCHWELCOME;
    }

    void CodContextInitializer::Uninitialize()
    {
        if (codec_context)
        {
            avcodec_free_context(&codec_context);
            codec_context = nullptr;
        }
    }

    VideoCodContextInitializer::VideoCodContextInitializer()
        : CodContextInitializer()
        , device_config(nullptr)
        , device(nullptr)
    {
    }

    VideoCodContextInitializer::~VideoCodContextInitializer()
    {
    }

    int VideoCodContextInitializer::Initialize(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options)
    {
        int res = 0;

		AVDictionary* param = nullptr;
        do
        {
            // find encoder by name first
            char codec_name[64] = { 0 };
            sprintf(codec_name, "%s_%s", avcodec_get_name(codec_id), AV_HWDEVICE_TYPE_CUDA == device_type ? "nvenc" : av_hwdevice_get_type_name(device_type));

            // if not found, then check to if have hardware supported
            if (!(codec = avcodec_find_encoder_by_name(codec_name)))
            {
                codec = avcodec_find_encoder(codec_id);
                if (!codec)
                {
                    res = AVERROR_DECODER_NOT_FOUND;
                    break;
                }
            }

            device_config = nullptr;
            if (AV_HWDEVICE_TYPE_NONE != device_type)
            {
                for (int i = 0; i <= 32; ++i)
                {
                    const AVCodecHWConfig* config = nullptr;
                    if ((config = avcodec_get_hw_config(codec, i)))
                    {
                        if (config->device_type == device_type && (config->methods & initialize_method) == initialize_method)
                        {
                            device_config = config;
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }

                if (!device_config)
                {
					res = AVERROR(ENODEV);
                    break;
                }
            }

            if (!(codec_context = avcodec_alloc_context3(codec)))
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if ((flags & AVFMT_GLOBALHEADER) == AVFMT_GLOBALHEADER)
            {
                codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            if (AV_HWDEVICE_TYPE_NONE != device_type)
            {
                // set device property
                char device_id[8] = { 0 };
                if (device_index >= 0)
                {
                    sprintf(device_id, "%d", device_index);
                }
                else
                {
                    sprintf(device_id, "auto");
                }

				for (std::map<std::string, std::string>::const_iterator optr = device_options.cbegin(); optr != device_options.cend() && res >= 0; ++optr)
				{
					res = av_dict_set(&param, optr->first.c_str(), optr->second.c_str(), 0);
				}
				if (res < 0)
				{
					break;
				}

                // create device
                if ((res = av_hwdevice_ctx_create(&device, device_type, device_id, param, 0)) < 0)
                {
                    break;
                }

                // set device context
                codec_context->hw_device_ctx = av_buffer_ref(device);

                // pass device configuration to callback
                codec_context->opaque = (void*)device_config;
            }
		} while (false);

		if (param)
		{
			av_dict_free(&param);
		}

        return res;
    }

    void VideoCodContextInitializer::Uninitialize()
    {
        CodContextInitializer::Uninitialize();

        if (device)
        {
            av_buffer_unref(&device);
        }
    }

    // --------- audio codec context implement -------------
    AudioCodContextInitializer::AudioCodContextInitializer()
        : CodContextInitializer()
    {
    }

    AudioCodContextInitializer::~AudioCodContextInitializer()
    {
    }

    int AudioCodContextInitializer::Initialize(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options)
    {
        (void)device_type;
        (void)device_index;
        (void)initialize_method;
		(void)device_options;

        int res = 0;
        do
        {
            if (!(codec = avcodec_find_encoder(codec_id)))
            {
                res = AVERROR_DECODER_NOT_FOUND;
                break;
            }

            if (!(codec_context = avcodec_alloc_context3(codec)))
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if ((flags & AVFMT_GLOBALHEADER) == AVFMT_GLOBALHEADER)
            {
                codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }
        } while (false);

        return res;
    }

    // --------- decoder implement -------------
    Encoder::Encoder(enum AVMediaType media_type)
        : encode_context(nullptr)
        , _packet(av_packet_alloc())
        , _frame(av_frame_alloc())
    {
        switch (media_type)
        {
        case AVMEDIA_TYPE_VIDEO:
        {
            encode_context = new VideoCodContextInitializer();
            break;
        }
        case AVMEDIA_TYPE_AUDIO:
        {
            encode_context = new AudioCodContextInitializer();
            break;
        }
        case AVMEDIA_TYPE_DATA:
            break;
        case AVMEDIA_TYPE_SUBTITLE:
            break;
        case AVMEDIA_TYPE_ATTACHMENT:
            break;
        default:
            break;
        }
    }

    Encoder::~Encoder()
    {
        if (encode_context)
        {
            delete encode_context;
            encode_context = nullptr;
        }
        if (_packet)
        {
            av_packet_free(&_packet);
        }
        if (_frame)
        {
            av_frame_free(&_frame);
        }
    }

    int Encoder::Create(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, void* context_initializer, int initialize_method, void* userdata, const std::map<std::string, std::string>& codec_options, const std::map<std::string, std::string>& device_options)
    {
        int res = 0;

        AVDictionary *param = nullptr;
        do
        {
            // frames context is used currently
            initialize_method = AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX;

            if (!encode_context)
            {
                res = AVERROR_STREAM_NOT_FOUND;
                break;
            }

            if ((res = encode_context->Initialize(codec_id, flags, device_type, device_index, initialize_method, device_options)) < 0)
            {
                break;
            }

            for (std::map<std::string, std::string>::const_iterator optr = codec_options.cbegin(); optr != codec_options.cend() && res >= 0; ++optr)
            {
                res = av_dict_set(&param, optr->first.c_str(), optr->second.c_str(), 0);
            }
            if (res < 0)
            {
                break;
            }

            if (context_initializer && (res = ((EncodeContextInitializer)context_initializer)(encode_context->codec_context, userdata)) < 0)
            {
                break;
            }

            res = avcodec_open2(encode_context->codec_context, encode_context->codec, &param);
        } while (false);

        if (param)
        {
            av_dict_free(&param);
        }

        return res;
    }

    int Encoder::Encode(const AVFrame* frame, const std::function<void(AVPacket*)>& encode_callback)
    {
        int res = 0;
        do 
        {
            const AVFrame* real_frame = frame;
            // hardware acceleration(frames context is used currently)
            if (encode_context->codec_context->hw_frames_ctx != frame->hw_frames_ctx)
            {
				// frame(buffer) is not allocated yet
                if (_frame->width <= 0 || _frame->height <= 0)
                {
                    _frame->width = frame->width;
                    _frame->height = frame->height;
                    _frame->format = frame->format;

                    if (encode_context->codec_context->hw_frames_ctx)
					{
						if ((res = av_hwframe_get_buffer(encode_context->codec_context->hw_frames_ctx, _frame, 0)) < 0)
						{
							break;
						}
                    } 
                    else
					{
						if ((res = av_frame_get_buffer(_frame, 1)) < 0)
						{
							break;
						}
                    }
                }

				// transfer data to hardware surface
				if ((res = av_hwframe_transfer_data(_frame, frame, 0)) < 0)
				{
					break;
				}

                // copy timestamp information
                _frame->pts = frame->pts;
                _frame->pkt_dts = frame->pkt_dts;
                _frame->pkt_duration = frame->pkt_duration;
                _frame->time_base = frame->time_base;
                _frame->opaque = frame->opaque;

                real_frame = _frame;
            }

            res = avcodec_send_frame(encode_context->codec_context, real_frame);
            if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
            {
                res = 0;
                break;
            }
            else
            {
                // encode all frames until there is no frame left
                while (res >= 0)
                {
                    // encode frame
                    res = avcodec_receive_packet(encode_context->codec_context, _packet);
                    if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
                    {
                        res = 0;
                        break;
                    }
                    else if (res < 0)
                    {
                        break;
                    }

                    _packet->duration = frame->pkt_duration;

                    // pass through the opaque
                    _packet->opaque = frame->opaque;

                    // call encode callback
                    encode_callback(_packet);
                }
            }
        } while (false);

        return res;
    }

	int Encoder::Encode(const AVFrame* frame, std::list<AVPacket*>& packets, size_t& size)
	{
		int res = 0;
		do
		{
			packets.clear();
			size = 0;

			const AVFrame* real_frame = frame;
			// hardware acceleration(frames context is used currently)
			if (encode_context->codec_context->hw_frames_ctx)
			{
				// frame(buffer) is not allocated yet
				if (!_frame->hw_frames_ctx)
				{
					_frame->width = frame->width;
					_frame->height = frame->height;
					_frame->format = frame->format;

					if ((res = av_hwframe_get_buffer(encode_context->codec_context->hw_frames_ctx, _frame, 0)) < 0)
					{
						break;
					}
				}

				// transfer data to hardware surface
				if ((res = av_hwframe_transfer_data(_frame, frame, 0)) < 0)
				{
					break;
				}

				// copy timestamp information
				_frame->pts = frame->pts;
				_frame->pkt_dts = frame->pkt_dts;
				_frame->pkt_duration = frame->pkt_duration;
				_frame->time_base = frame->time_base;
				_frame->opaque = frame->opaque;

				real_frame = _frame;
			}

			res = avcodec_send_frame(encode_context->codec_context, real_frame);
			if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
			{
				res = 0;
				break;
			}
			else
			{
				// encode all frames until there is no frame left
				while (res >= 0)
				{
					// encode frame
					AVPacket* packet = av_packet_alloc();
					res = avcodec_receive_packet(encode_context->codec_context, packet);
					if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
					{
						av_packet_free(&packet);
						res = 0;
						break;
					}
					else if (res < 0)
					{
						av_packet_free(&packet);
						break;
					}

					packet->duration = frame->pkt_duration;

					// pass through the opaque
					packet->opaque = frame->opaque;

					// add packet
					++size;
					packets.emplace_back(packet);
				}
			}
		} while (false);

		return res;
	}

	void Encoder::Destroy()
    {
		if (encode_context)
		{
			encode_context->Uninitialize();
		}
    }
};

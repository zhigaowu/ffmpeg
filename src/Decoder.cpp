
#pragma warning (disable:4819)

#include "Decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext_qsv.h>

#ifdef __cplusplus
}
#endif

#include <functional>

namespace ffmpeg
{
    enum AVPixelFormat frames_decode_context_initialize(AVCodecContext* ctx, const enum AVPixelFormat *pix_fmts)
    {
        const AVCodecHWConfig* device_config = static_cast<const AVCodecHWConfig*>(ctx->opaque);

        for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
        {
            if (*p == device_config->pix_fmt)
            {
                ctx->pix_fmt = device_config->pix_fmt;

				AVHWFramesConstraints* constraint = av_hwdevice_get_hwframe_constraints(ctx->hw_device_ctx, device_config);
                /* create a pool of surfaces to be used by the decoder */
                ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->hw_device_ctx);
                if (ctx->hw_frames_ctx)
                {
                    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)ctx->hw_frames_ctx->data;

					frames_ctx->format = device_config->pix_fmt;
					frames_ctx->sw_format = ctx->sw_pix_fmt;
                    frames_ctx->width = FFALIGN(ctx->width, 32);
                    frames_ctx->height = FFALIGN(ctx->height, 32);
					frames_ctx->initial_pool_size = 10;

					// choose software pixel format by constrain
					const AVPixFmtDescriptor* src_sw_pix_desc = av_pix_fmt_desc_get(ctx->sw_pix_fmt);
					if (constraint)
					{
						for (enum AVPixelFormat* sw_fmt = constraint->valid_sw_formats; sw_fmt && *sw_fmt != AV_PIX_FMT_NONE; ++sw_fmt)
						{
							const AVPixFmtDescriptor* dst_sw_pix_desc = av_pix_fmt_desc_get(*sw_fmt);
							if (src_sw_pix_desc->nb_components == dst_sw_pix_desc->nb_components)
							{
								uint8_t comp_index = 0;
								while (comp_index < src_sw_pix_desc->nb_components)
								{
									if (src_sw_pix_desc->comp[comp_index].plane != dst_sw_pix_desc->comp[comp_index].plane || src_sw_pix_desc->comp[comp_index].depth != dst_sw_pix_desc->comp[comp_index].depth)
									{
										break;
									}
									++comp_index;
								}

								if (comp_index >= src_sw_pix_desc->nb_components)
								{
									frames_ctx->sw_format = *sw_fmt;
									break;
								}
							}
						}
					}

                    if (av_hwframe_ctx_init(ctx->hw_frames_ctx) < 0)
					{
						av_hwframe_constraints_free(&constraint);
                        av_buffer_unref(&(ctx->hw_frames_ctx));
                        break;
                    }
                }
				av_hwframe_constraints_free(&constraint);

                return *p;
            }
        }
        return AV_PIX_FMT_NONE;
    }

    enum AVPixelFormat device_decode_context_initialize(AVCodecContext* ctx, const enum AVPixelFormat *pix_fmts)
    {
        const AVCodecHWConfig* device_config = static_cast<const AVCodecHWConfig*>(ctx->opaque);

        for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
        {
            if (*p == device_config->pix_fmt)
            {
                return *p;
            }
        }
        return AV_PIX_FMT_NONE;
    }

    DecContextInitializer::DecContextInitializer()
        : codec(nullptr), codec_context(nullptr)
    {
    }

    DecContextInitializer::~DecContextInitializer()
    {
    }

    int DecContextInitializer::Initialize(AVFormatContext*, enum AVHWDeviceType, int, int, const std::map<std::string, std::string>&)
    {
        return AVERROR_PATCHWELCOME;
    }

    void DecContextInitializer::Uninitialize()
    {
        if (codec_context)
        {
            avcodec_free_context(&codec_context);
            codec_context = nullptr;
        }
    }

    // --------- video codec context implement -------------
    VideoDecContextInitializer::VideoDecContextInitializer()
        : DecContextInitializer()
        , device_config(nullptr)
        , device(nullptr)
    {
    }

    VideoDecContextInitializer::~VideoDecContextInitializer()
    {
    }

    int VideoDecContextInitializer::Initialize(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options)
    {
        int res = 0;

		AVDictionary* param = nullptr;
        do 
        {
            AVStream* stream = nullptr;
            for (unsigned int i = 0; i < format_context->nb_streams; ++i)
            {
                if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                {
                    stream = format_context->streams[i];
                    break;
                }
            }

			// find decoder by name first
			char codec_name[64] = { 0 };
			sprintf(codec_name, "%s_%s", avcodec_get_name(stream->codecpar->codec_id), AV_HWDEVICE_TYPE_CUDA == device_type ? "nvdec" : av_hwdevice_get_type_name(device_type));

			// if not found, then check to if have hardware supported
			if (!(codec = avcodec_find_decoder_by_name(codec_name)))
			{
				if (!(codec = avcodec_find_decoder(stream->codecpar->codec_id)))
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

            if ((res = avcodec_parameters_to_context(codec_context, stream->codecpar)) < 0)
            {
                break;
            }
            codec_context->framerate = av_guess_frame_rate(format_context, stream, nullptr);
            codec_context->time_base = av_inv_q(codec_context->framerate);

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

    void VideoDecContextInitializer::Uninitialize()
    {
        DecContextInitializer::Uninitialize();

        if (device)
        {
            av_buffer_unref(&device);
        }
    }

    // --------- audio codec context implement -------------
    AudioDecContextInitializer::AudioDecContextInitializer()
        : DecContextInitializer()
    {
    }

    AudioDecContextInitializer::~AudioDecContextInitializer()
    {
    }

    int AudioDecContextInitializer::Initialize(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options)
    {
        (void)device_type;
        (void)device_index;
        (void)initialize_method;
		(void)device_options;

        int res = 0;
        do
        {
            AVStream* stream = nullptr;
            for (unsigned int i = 0; i < format_context->nb_streams; ++i)
            {
                if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
                {
                    stream = format_context->streams[i];
                    break;
                }
            }

            if (!(codec = avcodec_find_decoder(stream->codecpar->codec_id)))
            {
                res = AVERROR_DECODER_NOT_FOUND;
                break;
            }

            if (!(codec_context = avcodec_alloc_context3(codec)))
            {
				res = AVERROR(ENOMEM);
                break;
            }

            if ((res = avcodec_parameters_to_context(codec_context, stream->codecpar)) < 0)
            {
                break;
            }
        } while (false);

        return res;
    }

    // --------- decoder implement -------------
    Decoder::Decoder(enum AVMediaType media_type)
        : decode_context(nullptr)
        , _frame(av_frame_alloc())
    {
        switch (media_type)
        {
        case AVMEDIA_TYPE_VIDEO:
        {
            decode_context = new VideoDecContextInitializer();
            break;
        }
        case AVMEDIA_TYPE_AUDIO:
        {
            decode_context = new AudioDecContextInitializer();
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

    Decoder::~Decoder()
    {
        if (decode_context)
        {
            delete decode_context;
            decode_context = nullptr;
        }
        if (_frame)
        {
            av_frame_free(&_frame);
            _frame = nullptr;
        }
    }

    int Decoder::Create(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, void* context_initializer, int initialize_method, void* userdata, const std::map<std::string, std::string>& codec_options, const std::map<std::string, std::string>& device_options)
    {
        int res = 0;

        AVDictionary *param = nullptr;
        do
        {
            if (!decode_context)
            {
                res = AVERROR_STREAM_NOT_FOUND;
                break;
            }

            if ((res = decode_context->Initialize(format_context, device_type, device_index, initialize_method, device_options)) < 0)
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

            if (context_initializer && (res = ((DecodeContextInitializer)context_initializer)(decode_context->codec_context, userdata)) < 0)
            {
                break;
            }

            res = avcodec_open2(decode_context->codec_context, decode_context->codec, &param);
        } while (false);

        if (param)
        {
            av_dict_free(&param);
        }

        return res;
    }

    int Decoder::Decode(const AVPacket* packet, const std::function<void(AVFrame*)>& decode_callback, int64_t start_pts)
    {
        int res = avcodec_send_packet(decode_context->codec_context, packet);
        do 
        {
            if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
            {
                res = 0;
                break;
            }
            else
            {
                // decode all frames until there is no frame left
                while (res >= 0)
                {
                    // decode frame
                    res = avcodec_receive_frame(decode_context->codec_context, _frame);
                    if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
                    {
                        res = 0;
                        break;
                    }
                    else if (res < 0)
                    {
                        break;
                    }

                    if (packet->pts >= start_pts && decode_callback)
                    {
                        // pass through the opaque
                        _frame->opaque = packet->opaque;

                        // call decode callback
                        decode_callback(_frame);
                    }
                }
            }
        } while (false);

        return res;
    }

	int Decoder::Decode(const AVPacket* packet, std::list<AVFrame*>& frames, size_t& size, const std::function<void(AVFrame*)>& frame_callback, int64_t start_pts /*= 0LL*/)
	{
		int res = avcodec_send_packet(decode_context->codec_context, packet);
		do
		{
			frames.clear();
			size = 0;

			if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
			{
				res = 0;
				break;
			}
			else
			{
				// decode all frames until there is no frame left
				while (res >= 0)
				{
					// allocate frame
					AVFrame* frame = av_frame_alloc();

					// decode frame
					res = avcodec_receive_frame(decode_context->codec_context, frame);
					if (AVERROR(EAGAIN) == res || AVERROR_EOF == res)
					{
						av_frame_free(&frame);
						res = 0;
						break;
					}
					else if (res < 0)
					{
						av_frame_free(&frame);
						break;
					}

					if (packet->pts >= start_pts)
					{
						// pass through the opaque
						frame->opaque = packet->opaque;

						if (frame_callback)
						{
							frame_callback(frame);
						}

						++size;
						frames.push_back(frame);
					}
				}
			}
		} while (false);

		return res;
	}

	void Decoder::Destroy()
    {
        if (decode_context)
        {
            decode_context->Uninitialize();
        }
    }

};

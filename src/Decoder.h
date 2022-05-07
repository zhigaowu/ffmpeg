
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

#ifndef _FFMPEG_DECODER_HEADER_H_
#define _FFMPEG_DECODER_HEADER_H_

#ifdef __cplusplus
extern "C" {
#endif
 
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#ifdef __cplusplus
}
#endif

#include <string>
#include <list>
#include <map>

#include <functional>

namespace ffmpeg
{
    typedef int(*DecodeContextInitializer)(AVCodecContext* codec_context, void* userdata);

    enum AVPixelFormat frames_decode_context_initialize(AVCodecContext* ctx, const enum AVPixelFormat *pix_fmts);
    enum AVPixelFormat device_decode_context_initialize(AVCodecContext* ctx, const enum AVPixelFormat *pix_fmts);

    class DecContextInitializer
    {
    public:
        const AVCodec* codec;
        AVCodecContext* codec_context;

    public:
        explicit DecContextInitializer();
        virtual ~DecContextInitializer();
        
        virtual int Initialize(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options);
        virtual void Uninitialize();

    private:
        DecContextInitializer(DecContextInitializer& rhs) = delete;
        DecContextInitializer& operator=(DecContextInitializer& rhs) = delete;

        DecContextInitializer(DecContextInitializer&& rhs) = delete;
        DecContextInitializer& operator=(DecContextInitializer&& rhs) = delete;
    };

    class VideoDecContextInitializer : public DecContextInitializer
    {
    public:
        const AVCodecHWConfig* device_config;

    public:
        AVBufferRef* device;

    public:
        explicit VideoDecContextInitializer();
        ~VideoDecContextInitializer();

        int Initialize(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options) override;
        void Uninitialize() override;

    private:
        VideoDecContextInitializer(VideoDecContextInitializer& rhs) = delete;
        VideoDecContextInitializer& operator=(VideoDecContextInitializer& rhs) = delete;

        VideoDecContextInitializer(VideoDecContextInitializer&& rhs) = delete;
        VideoDecContextInitializer& operator=(VideoDecContextInitializer&& rhs) = delete;
    };

    class AudioDecContextInitializer : public DecContextInitializer
    {
    public:
        explicit AudioDecContextInitializer();
        ~AudioDecContextInitializer();

        int Initialize(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options) override;

    private:
        AudioDecContextInitializer(AudioDecContextInitializer& rhs) = delete;
        AudioDecContextInitializer& operator=(AudioDecContextInitializer& rhs) = delete;

        AudioDecContextInitializer(AudioDecContextInitializer&& rhs) = delete;
        AudioDecContextInitializer& operator=(AudioDecContextInitializer&& rhs) = delete;
    };

    class Decoder
    {
    public:
        DecContextInitializer* decode_context;

    public:
        explicit Decoder(enum AVMediaType media_type);
        ~Decoder();

        int Create(AVFormatContext* format_context, enum AVHWDeviceType device_type, int device_index, void* context_initializer, int initialize_method = AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX, void* userdata = nullptr, const std::map<std::string, std::string>& codec_options = std::map<std::string, std::string>(), const std::map<std::string, std::string>& device_options = std::map<std::string, std::string>());
        
        int Decode(const AVPacket* packet, const std::function<void(const AVPacket*, AVFrame*)>& decode_callback, int64_t start_pts = 0LL);

		int Decode(const AVPacket* packet, std::list<AVFrame*>& frames, size_t& size, const std::function<void(AVFrame*)>& frame_callback, int64_t start_pts = 0LL);

        void Destroy();

    private:
        AVFrame* _frame;

    private:
        Decoder() = delete;

        Decoder(Decoder& rhs) = delete;
        Decoder& operator=(Decoder& rhs) = delete;

        Decoder(Decoder&& rhs) = delete;
        Decoder& operator=(Decoder&& rhs) = delete;
    };
};

#endif

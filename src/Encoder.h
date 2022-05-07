
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

#ifndef _FFMPEG_ENCODER_HEADER_H_
#define _FFMPEG_ENCODER_HEADER_H_

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
#include <vector>
#include <map>

#include <functional>

namespace ffmpeg
{
    typedef int(*EncodeContextInitializer)(AVCodecContext* codec_context, void* userdata);

    int frames_encode_context_initialize(AVCodecContext* ctx, enum AVPixelFormat format);

    class CodContextInitializer
    {
    public:
        const AVCodec* codec;
        AVCodecContext* codec_context;

    public:
        explicit CodContextInitializer();
        virtual ~CodContextInitializer();
        
        virtual int Initialize(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options);
        virtual void Uninitialize();

    private:
        CodContextInitializer(CodContextInitializer& rhs) = delete;
        CodContextInitializer& operator=(CodContextInitializer& rhs) = delete;

        CodContextInitializer(CodContextInitializer&& rhs) = delete;
        CodContextInitializer& operator=(CodContextInitializer&& rhs) = delete;
    };

    class VideoCodContextInitializer : public CodContextInitializer
    {
    public:
        const AVCodecHWConfig* device_config;

    public:
        AVBufferRef* device;

    public:
        explicit VideoCodContextInitializer();
        ~VideoCodContextInitializer();

        int Initialize(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options) override;
        void Uninitialize() override;

    private:
        VideoCodContextInitializer(VideoCodContextInitializer& rhs) = delete;
        VideoCodContextInitializer& operator=(VideoCodContextInitializer& rhs) = delete;

        VideoCodContextInitializer(VideoCodContextInitializer&& rhs) = delete;
        VideoCodContextInitializer& operator=(VideoCodContextInitializer&& rhs) = delete;
    };

    class AudioCodContextInitializer : public CodContextInitializer
    {
    public:
        explicit AudioCodContextInitializer();
        ~AudioCodContextInitializer();

        int Initialize(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, int initialize_method, const std::map<std::string, std::string>& device_options) override;

    private:
        AudioCodContextInitializer(AudioCodContextInitializer& rhs) = delete;
        AudioCodContextInitializer& operator=(AudioCodContextInitializer& rhs) = delete;

        AudioCodContextInitializer(AudioCodContextInitializer&& rhs) = delete;
        AudioCodContextInitializer& operator=(AudioCodContextInitializer&& rhs) = delete;
    };

    class Encoder
    {
    public:
        CodContextInitializer* encode_context;

    public:
        explicit Encoder(enum AVMediaType media_type);
        ~Encoder();

        int Create(enum AVCodecID codec_id, int flags, enum AVHWDeviceType device_type, int device_index, void* context_initializer, int initialize_method = AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX, void* userdata = nullptr, const std::map<std::string,std::string>& codec_options = std::map<std::string, std::string>(), const std::map<std::string, std::string>& device_options = std::map<std::string, std::string>());
        
        int Encode(const AVFrame* frame, const std::function<void(AVPacket*)>& encode_callback);

		int Encode(const AVFrame* frame, std::list<AVPacket*>& packets, size_t& size);

        void Destroy();

    private:
        AVPacket* _packet;

    private:
        AVFrame* _frame;

    private:
        Encoder() = delete;

        Encoder(Encoder& rhs) = delete;
        Encoder& operator=(Encoder& rhs) = delete;

        Encoder(Encoder&& rhs) = delete;
        Encoder& operator=(Encoder&& rhs) = delete;
    };
};

#endif

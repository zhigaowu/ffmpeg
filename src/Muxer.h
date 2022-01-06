
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

#ifndef _FFMPEG_MUXER_HEADER_H_
#define _FFMPEG_MUXER_HEADER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>

#ifdef __cplusplus
}
#endif

#include <map>
#include <vector>
#include <string>

namespace ffmpeg
{
    class Muxer
    {
    public:
        AVFormatContext* format_context;
        AVStream **streams;

    public:
        int video_stream_index;
        const AVCodecParameters* video_codecpar;

    public:
        int audio_stream_index;
        const AVCodecParameters* audio_codecpar;

    public:
        explicit Muxer(const std::string& url, const char* format_name = nullptr, const AVOutputFormat *oformat = nullptr);
        ~Muxer();

        int AddStream(const AVCodec* codec, const AVCodecContext* codec_context);

        int AddStream(const AVStream* stream);

        int AddStream(const AVFormatContext* input_context);

        int Open(int64_t connect_timeout, int64_t read_timeout, const std::map<std::string, std::string>& options = std::map<std::string, std::string>());

        int Write(const AVPacket* packet);

        bool Timeout();

        int Close();

    private:
        bool checkStream(const AVStream* stream);

        void parseParameters();

    private:
        std::string _url;
		const char* _format_name;
		const AVOutputFormat* _oformat;

    private:
        std::vector<int> _stream_indexs;

    private:
        int64_t _write_timeout;
        int64_t _write_at;

    private:
        Muxer() = delete;

        Muxer(Muxer& rhs) = delete;
        Muxer& operator=(Muxer& rhs) = delete;

        Muxer(Muxer&& rhs) = delete;
        Muxer& operator=(Muxer&& rhs) = delete;
    };
};

#endif











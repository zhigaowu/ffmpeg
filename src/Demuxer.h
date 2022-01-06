
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

#ifndef _FFMPEG_DEMUXER_HEADER_H_
#define _FFMPEG_DEMUXER_HEADER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif

#include <map>
#include <string>

namespace ffmpeg
{
    class Demuxer
    {
	public:
		static inline bool IsVideo(const AVPacket* packet)
		{
			return (int)(packet->opaque) == AVMEDIA_TYPE_VIDEO;
		}

		static inline bool KeyPacket(const AVPacket* packet)
		{
			return (packet->flags & AV_PKT_FLAG_KEY) == AV_PKT_FLAG_KEY;
		}

    public:
        AVFormatContext* format_context;
        AVStream **streams;

    public:
        int64_t seek_pts;

    public:
        int video_stream_index;
        const AVCodecParameters* video_codecpar;

        double frame_rate;

    public:
        int audio_stream_index;
        const AVCodecParameters* audio_codecpar;

    public:
        explicit Demuxer();
        virtual ~Demuxer();

        int Open(int64_t connect_timeout, int64_t read_timeout, const std::map<std::string, std::string>& options = std::map<std::string, std::string>(), int64_t seek_micros = 0LL);

        int Seek(int64_t seek_micros);

        int Read(AVPacket** packet);

		bool Local();

        void Close();

    protected:
        virtual int open(const AVInputFormat** format, AVDictionary** param) = 0;

    private:
        void parseParameters();

	private:
		bool timeOut();
		friend int InterruptCb(void* userdata);

    private:
        int64_t _read_timeout;
        int64_t _read_at;

    private:
        Demuxer(Demuxer& rhs) = delete;
        Demuxer& operator=(Demuxer& rhs) = delete;

        Demuxer(Demuxer&& rhs) = delete;
        Demuxer& operator=(Demuxer&& rhs) = delete;
    };

    class UrlDemuxer : public Demuxer
    {
    public:
        explicit UrlDemuxer(const std::string& url);
        ~UrlDemuxer();

    protected:
        int open(const AVInputFormat** format, AVDictionary** param) override;

    private:
        std::string encodeUrl(const std::string& url);

    private:
        std::string _url;

    private:
        UrlDemuxer() = delete;

        UrlDemuxer(UrlDemuxer& rhs) = delete;
        UrlDemuxer& operator=(UrlDemuxer& rhs) = delete;

        UrlDemuxer(UrlDemuxer&& rhs) = delete;
        UrlDemuxer& operator=(UrlDemuxer&& rhs) = delete;
    };

    class UsbDemuxer : public Demuxer
    {
    public:
        /*
           usb: usb name
           device: 
                  windows: dshow|gdigrab
                  linux: video4linux2
        */
        explicit UsbDemuxer(const std::string& usb, const std::string& device);
        ~UsbDemuxer();

    protected:
        int open(const AVInputFormat** format, AVDictionary** param) override;

    private:
        std::string _usb;

    private:
        std::string _device;

    private:
        UsbDemuxer() = delete;

        UsbDemuxer(UsbDemuxer& rhs) = delete;
        UsbDemuxer& operator=(UsbDemuxer& rhs) = delete;

        UsbDemuxer(UsbDemuxer&& rhs) = delete;
        UsbDemuxer& operator=(UsbDemuxer&& rhs) = delete;
    };

    class DesktopDemuxer : public Demuxer
    {
    public:
        /*
           desktop: 
                  windows: video=screen-capture-recorder|desktop
                  linux: :0.0+0,0|1

           device:
                  windows: dshow|gdigrab
                  linux: x11grab|avfoundation
        */
        explicit DesktopDemuxer(const std::string& desktop, const std::string& frame_rate, const std::string& device);
        ~DesktopDemuxer();

    protected:
        int open(const AVInputFormat** format, AVDictionary** param) override;

    private:
        std::string _desktop;
        std::string _frame_rate;

    private:
        std::string _device;

    private:
        DesktopDemuxer() = delete;

        DesktopDemuxer(DesktopDemuxer& rhs) = delete;
        DesktopDemuxer& operator=(DesktopDemuxer& rhs) = delete;

        DesktopDemuxer(DesktopDemuxer&& rhs) = delete;
        DesktopDemuxer& operator=(DesktopDemuxer&& rhs) = delete;
    };

    class RawDemuxer : public Demuxer
    {
    public:
        explicit RawDemuxer(int(*raw_data_input)(void*, unsigned char*, int), void* userdata);
        ~RawDemuxer();

    protected:
        int open(const AVInputFormat** format, AVDictionary** param) override;

    private:
        int(*_raw_data_input)(void*, unsigned char*, int);
        void* _userdata;

    private:
        AVIOContext* _io_context;

    private:
        RawDemuxer() = delete;

        RawDemuxer(RawDemuxer& rhs) = delete;
        RawDemuxer& operator=(RawDemuxer& rhs) = delete;

        RawDemuxer(RawDemuxer&& rhs) = delete;
        RawDemuxer& operator=(RawDemuxer&& rhs) = delete;
    };
};

#endif











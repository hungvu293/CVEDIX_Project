#ifndef READER_H
#define READER_H

#include <iostream>
#include <thread>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

class Reader {
public:
    Reader();
    ~Reader();
    bool isOpened = false;
    int open(const std::string& input_url, bool use_hw = false);
    int decodeFrame(cv::Mat& frame);
    void close();

private:
    void print_error(const char *msg, int err);

    AVFormatContext *fmt_ctx;
    AVCodecContext *dec_ctx;
    const AVCodec *dec;
    AVPacket *pkt;
    AVFrame *frame;
    AVFrame *sw_frame;
    struct SwsContext *sws_ctx;
    AVBufferRef *hw_device_ctx;
    int video_stream_idx;
    uint8_t *rgb_buf;
    int rgb_bufsize;
    std::string rtsp_url; // store RTSP link

    std::chrono::steady_clock::time_point lastFrameTime;
    const int targetIntervalMs = 150;
};

#endif // READER_H
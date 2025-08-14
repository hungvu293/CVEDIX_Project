#ifndef READER_H
#define READER_H

#include <iostream>
#include <thread>
#include <string>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>

// #include "RgaUtils.h"
// #include "im2d.hpp"
#include "rga_helper.hpp"

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
    static std::mutex decode_mutex;

private:
    void print_error(const char *msg, int err);
    // int rga_cvt_color(AVFrame* src_frame, cv::Mat& dst_mat);

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
    bool use_hw_accel;

    std::chrono::steady_clock::time_point lastFrameTime;
    const int targetIntervalMs = 1000;
};

#endif // READER_H
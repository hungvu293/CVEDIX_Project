#ifndef SERVER_H
#define SERVER_H

#include <iostream>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

class Server {
public:
    Server();
    ~Server();

    int open(const std::string& output_url, int width, int height, int fps, bool use_hw = false);
    int encodeFrame(const cv::Mat& inFrame);
    void close();

private:
    void print_error(const char *msg, int err);

    AVFormatContext *oformat_ctx;
    AVCodecContext *enc_ctx;
    const AVCodec *encoder;
    AVStream *stream;
    AVFrame *frame;
    AVPacket *pkt;
    struct SwsContext *sws_ctx;

    int frame_width;
    int frame_height;
    int64_t frame_pts;
    bool isOpened;
};

#endif // SERVER_H

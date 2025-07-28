#ifndef READER_H
#define READER_H

#include <string>

#include <opencv2/opencv.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/dict.h>
}

class Reader {
public:
    Reader();
    ~Reader();
    
    bool isOpened = false;

    int open(const std::string& rtspUrl);
    cv::Mat decodeFrame();
    void close();

private:
    AVFormatContext* pFormatContext = nullptr;
    AVCodecContext* pCodecContext = nullptr;
    const AVCodec* pCodec = nullptr;
    AVFrame* pFrame = nullptr;
    AVFrame* pFrameRGB = nullptr;
    SwsContext* swsContext = nullptr;
    uint8_t* buffer = nullptr;
    int videoStreamIndex = -1;

    cv::Mat convertAVFrameToMat(AVFrame* frame);
    enum AVPixelFormat getCurrentSwsFormat();

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
};
#endif // READER_H
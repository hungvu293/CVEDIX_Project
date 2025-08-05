#ifndef READER_H
#define READER_H

#include <string>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

class Reader {
public:
    Reader();
    ~Reader();
    
    bool isOpened = false;
    int open(const std::string& rtspUrl);
    int decodeFrame(cv::Mat& frame);
    void close();

private:
    AVFormatContext* pFormatContext = nullptr;
    AVCodecContext* pCodecContext = nullptr;
    const AVCodec* pCodec = nullptr;
    AVFrame* pFrame = nullptr;
    int videoStreamIndex = -1;
    
    // Conversion methods
    int convert_rgb_software(AVFrame* frame, cv::Mat& output);
    cv::Mat convertAVFrameToMat(AVFrame* frame);
    enum AVPixelFormat getCurrentSwsFormat();
    
    // Disable copy
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
};

#endif // READER_H
#ifndef READER_H
#define READER_H

#include <string>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext_drm.h>
#include <libdrm/drm_fourcc.h>
#include <rga/RgaApi.h>
#include <rga/rga.h>
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
    uint8_t* rgbBuffer = nullptr;
    int rgbBufferSize = 0;
    int videoStreamIndex = -1;

    cv::Mat convertAVFrameToMat(AVFrame* frame);
    enum AVPixelFormat getCurrentSwsFormat();
    int convert_rgb(AVFrame* frame, uint8_t* rgb_buf);

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
};

#endif // READER_H
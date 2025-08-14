#pragma once
#include <mutex>
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

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"

// Chỉ khai báo biến mutex, không định nghĩa
extern std::mutex rgaMutex;

// Khai báo hàm, không định nghĩa
int resize_rga(const cv::Mat &image, cv::Mat &resized_image);
int rga_cvt_color(AVFrame* src_frame, cv::Mat& dst_mat);
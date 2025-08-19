#pragma once
#include <mutex>
#include <opencv2/opencv.hpp>

extern "C" {
#include <dev/ffmpeg/libavformat/avformat.h>
#include <dev/ffmpeg/libavcodec/avcodec.h>
#include <dev/ffmpeg/libavutil/opt.h>
#include <dev/ffmpeg/libavutil/error.h>
#include <dev/ffmpeg/libavutil/imgutils.h>
#include <dev/ffmpeg/libavutil/hwcontext.h>
#include <dev/ffmpeg/libswscale/swscale.h>
}

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "dma_alloc.h"

#define DMA_HEAP_DMA32_UNCACHED_PATH "/dev/dma_heap/system-uncached-dma32"

static std::mutex dma_mutex;

int resize_rga(const cv::Mat &image, cv::Mat &resized_image);
int rga_cvt_color(AVFrame* src_frame, cv::Mat& dst_mat);
#include "reader.h" 
#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixfmt.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

enum AVPixelFormat get_format(AVCodecContext *Context, const enum AVPixelFormat *PixFmt)
{
    while (*PixFmt != AV_PIX_FMT_NONE) {
        if (*PixFmt == AV_PIX_FMT_DRM_PRIME)
            return AV_PIX_FMT_DRM_PRIME;
        PixFmt++;
    }
    return AV_PIX_FMT_NONE;
}

Reader::Reader() {
    avformat_network_init();
    av_log_set_level(AV_LOG_ERROR);
}

Reader::~Reader() {
    close();
}

int Reader::open(const std::string& rtspUrl) {
    if (isOpened) {
        std::cerr << "Reader already opened. Please close first." << std::endl;
        return -1;
    }
    
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "buffer_size", "2048000", 0);
    av_dict_set(&opts, "analyzeduration", "5000000", 0);
    av_dict_set(&opts, "probesize", "5000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "5000000", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);

    if (avformat_open_input(&pFormatContext, rtspUrl.c_str(), nullptr, &opts) < 0) {
        std::cerr << "Could not open RTSP stream: " << rtspUrl << std::endl;
        av_dict_free(&opts);
        return -1;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(pFormatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        close();
        return -1;
    }

    videoStreamIndex = -1;
    for (unsigned int i = 0; i < pFormatContext->nb_streams; i++) {
        if (pFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Did not find a video stream." << std::endl;
        close();
        return -1;
    }

    AVCodecParameters* pCodecPar = pFormatContext->streams[videoStreamIndex]->codecpar;
    
    // Try software decoder first to avoid hardware issues
    pCodec = avcodec_find_decoder(pCodecPar->codec_id);
    if (pCodec) {
        std::cout << "Using software decoder: " << pCodec->name << std::endl;
    } else {
        std::cerr << "Unsupported codec!" << std::endl;
        close();
        return -1;
    }

    pCodecContext = avcodec_alloc_context3(pCodec);
    if (!pCodecContext) {
        std::cerr << "Failed to allocate AVCodecContext." << std::endl;
        close();
        return -1;
    }

    if (avcodec_parameters_to_context(pCodecContext, pCodecPar) < 0) {
        std::cerr << "Failed to copy codec parameters to decoder context." << std::endl;
        close();
        return -1;
    }

    std::cout << "[open] pCodecContext: width=" << pCodecContext->width
              << ", height=" << pCodecContext->height
              << ", pix_fmt=" << pCodecContext->pix_fmt << std::endl;

    // Use software decoding to avoid DRM/RGA issues
    AVDictionary* codecOpts = nullptr;
    av_dict_set(&codecOpts, "threads", "auto", 0);

    if (avcodec_open2(pCodecContext, pCodec, &codecOpts) < 0) {
        std::cerr << "Failed to open codec." << std::endl;
        av_dict_free(&codecOpts);
        close();
        return -1;
    }
    av_dict_free(&codecOpts);

    pFrame = av_frame_alloc();
    if (!pFrame) {
        std::cerr << "Failed to allocate AVFrame." << std::endl;
        close();
        return -1;
    }

    isOpened = true;
    return 0;
}
static std::mutex sws_mutex;

int Reader::convert_rgb_software(AVFrame* frame, cv::Mat& output) {
    if (!frame) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(sws_mutex);
    static struct SwsContext* swsContext = nullptr;
    
    int width = frame->width;
    int height = frame->height;
    
    // Initialize swscale context
    swsContext = sws_getCachedContext(swsContext,
                                    width, height, (AVPixelFormat)frame->format,
                                    width, height, AV_PIX_FMT_BGR24,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
    
    if (!swsContext) {
        std::cerr << "Failed to initialize swscale context" << std::endl;
        return -1;
    }
    
    // Create output Mat
    output = cv::Mat(height, width, CV_8UC3);
    
    uint8_t* dst_data[4] = { output.data };
    int dst_linesize[4] = { static_cast<int>(output.step[0]) };
    
    // Convert
    sws_scale(swsContext, frame->data, frame->linesize,
              0, height, dst_data, dst_linesize);
    
    return 0;
}

int Reader::decodeFrame(cv::Mat& frame) {
    if (!isOpened) {
        std::cerr << "Reader is not opened. Call open() first." << std::endl;
        return -1;
    }

    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Failed to allocate packet" << std::endl;
        return -1;
    }

    int frameCount = 0;
    const int maxFrameAttempts = 10; 

    while (av_read_frame(pFormatContext, packet) >= 0 && frameCount < maxFrameAttempts) {
        frameCount++;
        
        if (packet->stream_index == videoStreamIndex) {
            int response = avcodec_send_packet(pCodecContext, packet);
            if (response < 0) {
                std::cerr << "Error while sending a packet to the decoder: " << response << std::endl;
                av_packet_unref(packet);
                continue;
            }

            while (response >= 0) {
                response = avcodec_receive_frame(pCodecContext, pFrame);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                    break;
                } else if (response < 0) {
                    std::cerr << "Error while receiving a frame from the decoder: " << response << std::endl;
                    break;
                }

                if (pFrame->width <= 0 || pFrame->height <= 0) {
                    std::cerr << "Invalid frame dimensions: " << pFrame->width << "x" << pFrame->height << std::endl;
                    continue;
                }

                // Use software conversion
                if (convert_rgb_software(pFrame, frame) == 0) {

                    av_packet_unref(packet);
                    av_packet_free(&packet);
                    return 0; // Successfully decoded and converted frame
                } else {
                    std::cerr << "Software conversion failed" << std::endl;
                    av_packet_unref(packet);
                    av_packet_free(&packet);
                    continue;
                }
            }
        }
        av_packet_unref(packet);
    }

    av_packet_free(&packet);
    
    if (frameCount >= maxFrameAttempts) {
        std::cerr << "Max frame attempts reached, stream might be problematic." << std::endl;
    }

    return -1; 
}

cv::Mat Reader::convertAVFrameToMat(AVFrame* frame) {
    int width = frame->width > 0 ? frame->width : pCodecContext->width;
    int height = frame->height > 0 ? frame->height : pCodecContext->height;
    
    cv::Mat mat(height, width, CV_8UC3, frame->data[0], frame->linesize[0]);
    return mat.clone();
}

enum AVPixelFormat Reader::getCurrentSwsFormat() {
    return (AVPixelFormat)pCodecContext->pix_fmt; 
}

void Reader::close() {
    if (pFrame) {
        av_frame_free(&pFrame);
        pFrame = nullptr;
    }
    if (pCodecContext) {
        avcodec_free_context(&pCodecContext);
        pCodecContext = nullptr;
    }
    if (pFormatContext) {
        avformat_close_input(&pFormatContext);
        pFormatContext = nullptr;
    }
    avformat_network_deinit(); 
    isOpened = false;
    std::cout << "Reader closed and resources freed." << std::endl;
}
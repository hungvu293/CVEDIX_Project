#include "reader.h" 

#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixfmt.h>
#include <rga/RgaApi.h>
#include <rga/rga.h>
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

static uint32_t drm_get_rgaformat(uint32_t drm_fmt)
{
    switch (drm_fmt) {
    case DRM_FORMAT_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case DRM_FORMAT_NV15:
        return RK_FORMAT_YCbCr_420_SP_10B;
    case DRM_FORMAT_NV16:
        return RK_FORMAT_YCbCr_422_SP;
    case DRM_FORMAT_YUYV:
        return RK_FORMAT_YUYV_422;
    case DRM_FORMAT_UYVY:
        return RK_FORMAT_UYVY_422;
    default:
        return 0;
    }
}

Reader::Reader() {
    avformat_network_init();
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
    
    const char* mppDecoderName = nullptr;
    switch (pCodecPar->codec_id) {
        case AV_CODEC_ID_H264:
            mppDecoderName = "h264_rkmpp";
            break;
        case AV_CODEC_ID_HEVC:
            mppDecoderName = "hevc_rkmpp";
            break;
        case AV_CODEC_ID_VP8:
            mppDecoderName = "vp8_rkmpp";
            break;
        case AV_CODEC_ID_VP9:
            mppDecoderName = "vp9_rkmpp";
            break;
        case AV_CODEC_ID_AV1:
            mppDecoderName = "av1_rkmpp";
            break;
        case AV_CODEC_ID_MJPEG:
            mppDecoderName = "mjpeg_rkmpp";
            break;
        case AV_CODEC_ID_MPEG1VIDEO:
            mppDecoderName = "mpeg1_rkmpp";
            break;
        case AV_CODEC_ID_MPEG2VIDEO:
            mppDecoderName = "mpeg2_rkmpp";
            break;
        case AV_CODEC_ID_MPEG4:
            mppDecoderName = "mpeg4_rkmpp";
            break;
        case AV_CODEC_ID_H263:
            mppDecoderName = "h263_rkmpp";
            break;
        default:
            std::cout << "Codec not supported by MPP, falling back to software decoder." << std::endl;
            break;
    }

    if (mppDecoderName) {
        pCodec = avcodec_find_decoder_by_name(mppDecoderName);
        if (pCodec) {
            std::cout << "Using MPP hardware decoder: " << mppDecoderName << std::endl;
        } else {
            std::cout << "MPP decoder not available, trying software decoder." << std::endl;
        }
    }
    
    if (!pCodec) {
        pCodec = avcodec_find_decoder(pCodecPar->codec_id);
        if (pCodec) {
            std::cout << "Using software decoder: " << pCodec->name << std::endl;
        }
    }

    if (!pCodec) {
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

    // Log thông số codec context
    std::cout << "[open] pCodecContext: width=" << pCodecContext->width
              << ", height=" << pCodecContext->height
              << ", pix_fmt=" << pCodecContext->pix_fmt
              << ", coded_width=" << pCodecContext->coded_width
              << ", coded_height=" << pCodecContext->coded_height
              << std::endl;

    // Set DRM PRIME format for hardware decoding
    pCodecContext->pix_fmt = AV_PIX_FMT_DRM_PRIME;
    pCodecContext->get_format = get_format;

    AVDictionary* codecOpts = nullptr;
    if (strstr(pCodec->name, "rkmpp")) {
        av_dict_set(&codecOpts, "async_depth", "1", 0);
    } else {
        av_dict_set(&codecOpts, "threads", "auto", 0);
    }

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

    // Allocate RGB buffer for conversion
    // int width = pCodecContext->width > 0 ? pCodecContext->width : 1920;
    // int height = pCodecContext->height > 0 ? pCodecContext->height : 1080;
    int width = pCodecContext->width;
    int height = pCodecContext->height;
    
    rgbBufferSize = width * height * 3; // RGB24
    rgbBuffer = (uint8_t*)av_malloc(rgbBufferSize);
    if (!rgbBuffer) {
        std::cerr << "Failed to allocate RGB buffer." << std::endl;
        close();
        return -1;
    }

    isOpened = true;
    std::cout << "Successfully opened RTSP stream with " 
              << (strstr(pCodec->name, "rkmpp") ? "hardware" : "software") 
              << " decoder." << std::endl;
    return 0;
}

int Reader::convert_rgb(AVFrame* frame, uint8_t* rgb_buf) {
    if (!frame || !rgb_buf) {
        std::cerr << "Invalid frame or buffer for RGB conversion." << std::endl;
        return -1;
    }

    AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor *)frame->data[0];
    if (!desc) {
        std::cerr << "No DRM frame descriptor available." << std::endl;
        return -1;
    }

    AVDRMLayerDescriptor *layer = &desc->layers[0];
    if (!layer) {
        std::cerr << "No DRM layer descriptor available." << std::endl;
        return -1;
    }

    // Get stride information
    int wStride = layer->planes[0].pitch;
    int hStride = (layer->planes[1].offset / layer->planes[0].pitch);
    
    // Get DRM format and convert to RGA format
    uint32_t drm_format = layer->format;
    RgaSURF_FORMAT src_format = (RgaSURF_FORMAT)drm_get_rgaformat(drm_format);
    
    if (src_format == 0) {
        std::cerr << "Unsupported DRM format: " << drm_format << std::endl;
        return -1;
    }

    // Setup RGA source
    rga_info_t src;
    memset(&src, 0, sizeof(rga_info_t));
    src.fd = desc->objects[0].fd;
    src.mmuFlag = 1;

    // Setup RGA destination  
    rga_info_t dst;
    memset(&dst, 0, sizeof(rga_info_t));
    dst.fd = -1;
    dst.virAddr = rgb_buf;
    dst.mmuFlag = 1;

    // Set rectangles for conversion (no resize, same dimensions)
    rga_set_rect(&src.rect, 0, 0, frame->width, frame->height, wStride, hStride, src_format);
    rga_set_rect(&dst.rect, 0, 0, frame->width, frame->height, frame->width, frame->height, RK_FORMAT_RGB_888);

    // Perform RGA conversion
    int ret = c_RkRgaBlit(&src, &dst, NULL);
    if (ret) {
        std::cerr << "RGA conversion failed with error: " << ret << std::endl;
        return -1;
    }

    return 0;
}

int Reader::decodeFrame(cv::Mat& frame) {
    if (!isOpened) {
        std::cerr << "Reader is not opened. Call open() first." << std::endl;
        return -1;
    }

    AVPacket packet;
    int frameCount = 0;
    const int maxFrameAttempts = 5; 

    while (av_read_frame(pFormatContext, &packet) >= 0 && frameCount < maxFrameAttempts) {
        frameCount++;
        
        if (packet.stream_index == videoStreamIndex) {
            int response = avcodec_send_packet(pCodecContext, &packet);
            if (response < 0) {
                std::cerr << "Error while sending a packet to the decoder: " << response << std::endl;
                av_packet_unref(&packet);
                close();
                return -1;
                // continue;
            }

            while (response >= 0) {
                response = avcodec_receive_frame(pCodecContext, pFrame);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                    // std::cerr << "averror" << std::endl;
                    // close();
                    // return -1;
                    break;
                } else if (response < 0) {
                    std::cerr << "Error while receiving a frame from the decoder: " << response << std::endl;
                    // close();
                    // return -1;
                    break;
                }

                if (pFrame->width <= 0 || pFrame->height <= 0) {
                    // std::cerr << "Skip empty frame." << std::endl;
                    // break;
                    std::cerr << "frame error" << std::endl;
                    close();
                    return -1;
                }

                // Check if we have a DRM PRIME frame
                if (pFrame->format == AV_PIX_FMT_DRM_PRIME) {
                    // Use RGA for hardware-accelerated conversion
                    if (convert_rgb(pFrame, rgbBuffer) == 0) {
                        // Create OpenCV Mat from RGB buffer
                        frame = cv::Mat(pFrame->height, pFrame->width, CV_8UC3, rgbBuffer).clone();
                        av_packet_unref(&packet);
                        return 0; // Successfully decoded and converted frame
                    } else {
                        std::cerr << "Failed to convert frame using RGA." << std::endl;
                        // break;
                        close();
                        return -1;
                    }
                } else {
                    std::cerr << "Frame format is not DRM_PRIME: " << pFrame->format << std::endl;
                    // break;
                    close();
                    return -1;
                }
            }
        }
        av_packet_unref(&packet);
    }

    if (frameCount >= maxFrameAttempts) {
        std::cerr << "Max frame attempts reached, stream might be problematic." << std::endl;
        close();
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
    return AV_PIX_FMT_DRM_PRIME; 
}

void Reader::close() {
    if (rgbBuffer) {
        av_free(rgbBuffer);
        rgbBuffer = nullptr;
    }
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
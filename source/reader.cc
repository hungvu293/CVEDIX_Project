#include "reader.h" 

#include <iostream>

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
    pFrameRGB = av_frame_alloc();
    if (!pFrame || !pFrameRGB) {
        std::cerr << "Failed to allocate AVFrame." << std::endl;
        close();
        return -1;
    }

    int width = pCodecContext->width;
    int height = pCodecContext->height;
    
    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid frame dimensions: " << width << "x" << height << std::endl;
        close();
        return -1;
    }

    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    if (!buffer) {
        std::cerr << "Failed to allocate buffer." << std::endl;
        close();
        return -1;
    }

    av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_RGB24,
                         width, height, 1);

    enum AVPixelFormat srcFormat = pCodecContext->pix_fmt;
    if (srcFormat == AV_PIX_FMT_NONE) {
        if (strstr(pCodec->name, "rkmpp")) {
            srcFormat = AV_PIX_FMT_NV12; 
        } else {
            srcFormat = AV_PIX_FMT_YUV420P;
        }
        std::cout << "Codec format unknown, assuming: " << av_get_pix_fmt_name(srcFormat) << std::endl;
    }

    swsContext = sws_getContext(
        width,
        height,
        srcFormat,
        width,
        height,
        AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr
    );

    if (!swsContext) {
        std::cerr << "Could not initialize SwsContext with format: " 
                  << av_get_pix_fmt_name(srcFormat) << std::endl;
        close();
        return -1;
    }

    isOpened = true;
    std::cout << "Successfully opened RTSP stream with " 
              << (strstr(pCodec->name, "rkmpp") ? "hardware" : "software") 
              << " decoder." << std::endl;
    return 0;
}

cv::Mat Reader::decodeFrame() {
    if (!isOpened) {
        std::cerr << "Reader is not opened. Call open() first." << std::endl;
        return cv::Mat();
    }

    AVPacket packet;
    cv::Mat frameMat;
    int frameCount = 0;
    const int maxFrameAttempts = 10; 

    while (av_read_frame(pFormatContext, &packet) >= 0 && frameCount < maxFrameAttempts) {
        frameCount++;
        
        if (packet.stream_index == videoStreamIndex) {
            int response = avcodec_send_packet(pCodecContext, &packet);
            if (response < 0) {
                std::cerr << "Error while sending a packet to the decoder: " << response << std::endl;
                av_packet_unref(&packet);
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

                if (pFrame->format != AV_PIX_FMT_NONE) {
                    enum AVPixelFormat frameFormat = (enum AVPixelFormat)pFrame->format;
                    
                    static bool formatLogged = false;
                    if (!formatLogged) {
                        std::cout << "First frame format: " << av_get_pix_fmt_name(frameFormat) << std::endl;
                        formatLogged = true;
                    }
                    
                    bool needRecreate = false;
                    
                    if (frameFormat != AV_PIX_FMT_NV12 && frameFormat != AV_PIX_FMT_YUV420P) {
                        needRecreate = true;
                    }
                    
                    if (needRecreate) {
                        std::cout << "Recreating SwsContext for format: " 
                                  << av_get_pix_fmt_name(frameFormat) << std::endl;
                        
                        if (swsContext) {
                            sws_freeContext(swsContext);
                        }
                        
                        swsContext = sws_getContext(
                            pFrame->width,
                            pFrame->height,
                            frameFormat,
                            pFrame->width,
                            pFrame->height,
                            AV_PIX_FMT_RGB24,
                            SWS_BILINEAR,
                            nullptr,
                            nullptr,
                            nullptr
                        );
                        
                        if (!swsContext) {
                            std::cerr << "Could not recreate SwsContext with format: " 
                                      << av_get_pix_fmt_name(frameFormat) << std::endl;
                            av_packet_unref(&packet);
                            return cv::Mat();
                        }
                    }
                }

                sws_scale(swsContext, pFrame->data, pFrame->linesize, 0, pFrame->height,
                          pFrameRGB->data, pFrameRGB->linesize);

                frameMat = convertAVFrameToMat(pFrameRGB);
                av_packet_unref(&packet);
                return frameMat;
            }
        }
        av_packet_unref(&packet);
    }

    if (frameCount >= maxFrameAttempts) {
        std::cerr << "Max frame attempts reached, stream might be problematic." << std::endl;
    }

    return cv::Mat();
}

cv::Mat Reader::convertAVFrameToMat(AVFrame* frame) {
    int width = frame->width > 0 ? frame->width : pCodecContext->width;
    int height = frame->height > 0 ? frame->height : pCodecContext->height;
    
    cv::Mat mat(height, width, CV_8UC3, frame->data[0], frame->linesize[0]);
    return mat.clone();
}

enum AVPixelFormat Reader::getCurrentSwsFormat() {
    return AV_PIX_FMT_NONE; 
}

void Reader::close() {
    if (swsContext) {
        sws_freeContext(swsContext);
        swsContext = nullptr;
    }
    if (buffer) {
        av_free(buffer);
        buffer = nullptr;
    }
    if (pFrameRGB) {
        av_frame_free(&pFrameRGB);
        pFrameRGB = nullptr;
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
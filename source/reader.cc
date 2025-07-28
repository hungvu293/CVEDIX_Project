#include "reader.h" 

#include <iostream>

Reader::Reader() {
    avformat_network_init();
}

Reader::~Reader() {
    close();
}

bool Reader::open(const std::string& rtspUrl) {
    if (isOpened) {
        std::cerr << "Reader already opened. Please close first." << std::endl;
        return false;
    }
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    av_dict_set(&opts, "analyzeduration", "5000000", 0);
    av_dict_set(&opts, "probesize", "5000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "5000000", 0);

    if (avformat_open_input(&pFormatContext, rtspUrl.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open RTSP stream: " << rtspUrl << std::endl;
        return false;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(pFormatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        close();
        return false;
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
        return false;
    }

    AVCodecParameters* pCodecPar = pFormatContext->streams[videoStreamIndex]->codecpar;

    pCodec = avcodec_find_decoder(pCodecPar->codec_id);
    if (!pCodec) {
        std::cerr << "Unsupported codec!" << std::endl;
        close();
        return false;
    }

    pCodecContext = avcodec_alloc_context3(pCodec);
    if (!pCodecContext) {
        std::cerr << "Failed to allocate AVCodecContext." << std::endl;
        close();
        return false;
    }

    if (avcodec_parameters_to_context(pCodecContext, pCodecPar) < 0) {
        std::cerr << "Failed to copy codec parameters to decoder context." << std::endl;
        close();
        return false;
    }

    if (avcodec_open2(pCodecContext, pCodec, nullptr) < 0) {
        std::cerr << "Failed to open codec." << std::endl;
        close();
        return false;
    }

    pFrame = av_frame_alloc();
    pFrameRGB = av_frame_alloc();
    if (!pFrame || !pFrameRGB) {
        std::cerr << "Failed to allocate AVFrame." << std::endl;
        close();
        return false;
    }

    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecContext->width, pCodecContext->height, 1);
    buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_RGB24,
                         pCodecContext->width, pCodecContext->height, 1);

    swsContext = sws_getContext(
        pCodecContext->width,
        pCodecContext->height,
        pCodecContext->pix_fmt,
        pCodecContext->width,
        pCodecContext->height,
        AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr
    );

    if (!swsContext) {
        std::cerr << "Could not initialize SwsContext." << std::endl;
        close();
        return false;
    }

    isOpened = true;
    return true;
}

cv::Mat Reader::decodeFrame() {
    if (!isOpened) {
        std::cerr << "Reader is not opened. Call open() first." << std::endl;
        return cv::Mat();
    }

    AVPacket packet;
    cv::Mat frameMat;

    while (av_read_frame(pFormatContext, &packet) >= 0) {
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

                sws_scale(swsContext, pFrame->data, pFrame->linesize, 0, pCodecContext->height,
                          pFrameRGB->data, pFrameRGB->linesize);

                frameMat = convertAVFrameToMat(pFrameRGB);
                av_packet_unref(&packet);
                return frameMat;
            }
        }
        av_packet_unref(&packet);
    }

    return cv::Mat();
}

cv::Mat Reader::convertAVFrameToMat(AVFrame* frame) {
    cv::Mat mat(pCodecContext->height, pCodecContext->width, CV_8UC3, frame->data[0], frame->linesize[0]);
    return mat.clone();
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
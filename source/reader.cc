#include "reader.h"

Reader::Reader() : fmt_ctx(nullptr), dec_ctx(nullptr), dec(nullptr), pkt(nullptr), frame(nullptr), sw_frame(nullptr),
                   sws_ctx(nullptr), hw_device_ctx(nullptr), video_stream_idx(-1), rgb_buf(nullptr), rgb_bufsize(0), isOpened(false), use_hw_accel(false) {}

Reader::~Reader() {
    close();
}

void Reader::print_error(const char *msg, int err) {
    char errbuf[128];
    av_strerror(err, errbuf, sizeof(errbuf));
    fprintf(stderr, "%s: %s\n", msg, errbuf);
}

int Reader::open(const std::string& input_url, bool use_hw) {
    // av_log_set_level(AV_LOG_DEBUG); // Keep debug logging for now
    av_log_set_level(AV_LOG_INFO);
    use_hw_accel = use_hw;

    int ret = 0;

    // Set options for RTSP with larger probesize and analyzeduration
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);         
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "fflags", "nobuffer+discardcorrupt", 0);
    av_dict_set(&opts, "flush_packets", "1", 0);
    av_dict_set(&opts, "probesize", "100000", 0);          
    av_dict_set(&opts, "analyzeduration", "100000", 0);   
    av_dict_set(&opts, "buffer_size", "1048576", 0);      
    av_dict_set(&opts, "reorder_queue_size", "3", 0);    

    // Open input stream
    if ((ret = avformat_open_input(&fmt_ctx, input_url.c_str(), NULL, &opts)) < 0) {
        print_error("Cannot open input", ret);
        av_dict_free(&opts);
        close();
        return ret;
    }
    av_dict_free(&opts);

    // Set format context flags
    fmt_ctx->avio_flags |= AVIO_FLAG_DIRECT;
    fmt_ctx->flags |= AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;
    fmt_ctx->flags |= AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;
    fmt_ctx->max_delay = 0;

    // Find stream info
    if ((ret = avformat_find_stream_info(fmt_ctx, NULL)) < 0) {
        print_error("Cannot find stream info", ret);
        close();
        return ret;
    }

    // Find video stream with safety checks
    video_stream_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (!fmt_ctx->streams[i] || !fmt_ctx->streams[i]->codecpar) {
            fprintf(stderr, "Stream %u has invalid or null codec parameters\n", i);
            continue;
        }
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx < 0) {
        fprintf(stderr, "No valid video stream found\n");
        close();
        return AVERROR_STREAM_NOT_FOUND;
    }

    // Decoder selection
    if (use_hw) {
        const char *prefer_decoder = "h264_rkmpp";
        dec = avcodec_find_decoder_by_name(prefer_decoder);
        if (dec) {
            fprintf(stdout, "Using hardware decoder: %s\n", prefer_decoder);
        } else {
            fprintf(stderr, "Hardware decoder %s not found, falling back to software decoder.\n", prefer_decoder);
            use_hw_accel = false;
        }
    }

    if (!dec) {
        dec = avcodec_find_decoder(fmt_ctx->streams[video_stream_idx]->codecpar->codec_id);
        if (!dec) {
            fprintf(stderr, "No suitable decoder found\n");
            close();
            return AVERROR_DECODER_NOT_FOUND;
        }
        fprintf(stdout, "Using software decoder: %s\n", dec->name);
    }

    // Allocate and configure decoder context
    dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) {
        close();
        return AVERROR(ENOMEM);
    }
    dec_ctx->thread_count = 1; // Avoid threading issues with RGA

    if ((ret = avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[video_stream_idx]->codecpar)) < 0) {
        print_error("Failed to copy codec parameters", ret);
        close();
        return ret;
    }

    // Open codec
    AVDictionary *codec_opts = NULL;
    av_dict_set(&codec_opts, "strict", "experimental", 0);
    if ((ret = avcodec_open2(dec_ctx, dec, &codec_opts)) < 0) {
        print_error("Failed to open codec", ret);
        av_dict_free(&codec_opts);
        close();
        return ret;
    }
    av_dict_free(&codec_opts);

    // Allocate packet and frames
    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    sw_frame = av_frame_alloc();
    if (!pkt || !frame || !sw_frame) {
        close();
        return AVERROR(ENOMEM);
    }

    rtsp_url = input_url;
    isOpened = true;
    lastFrameTime = std::chrono::steady_clock::now();
    fprintf(stdout, "Reader opened stream successfully from: %s\n", rtsp_url.c_str());
    return 0;
}



int Reader::decodeFrame(cv::Mat& outFrame) {
    int ret = 0;

    if ((ret = av_read_frame(fmt_ctx, pkt)) < 0) {
        close();
        return ret;
    }

    if (pkt->stream_index == video_stream_idx) {
        ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret < 0) {
            print_error("Error sending packet", ret);
            av_packet_unref(pkt);
            close();
            return ret;
        }

        while ((ret = avcodec_receive_frame(dec_ctx, frame)) >= 0) {
            AVFrame *convert_src = nullptr;
            if (frame->hw_frames_ctx || av_pix_fmt_desc_get((AVPixelFormat)frame->format)->flags & AV_PIX_FMT_FLAG_HWACCEL) {
                ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                if (ret < 0) {
                    print_error("Failed to transfer hw frame", ret);
                    av_frame_unref(frame);
                    av_frame_unref(sw_frame);
                    av_packet_unref(pkt);
                    close();
                }
                convert_src = sw_frame;
            } else {
                convert_src = frame;
            }

            // Frame skipping logic
            auto now = std::chrono::steady_clock::now();
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();
            if (elapsedMs < targetIntervalMs) {
                av_frame_unref(frame);
                continue;
            }
            lastFrameTime = now;

            if (!convert_src || convert_src->width <= 0 || convert_src->height <= 0 || convert_src->format != AV_PIX_FMT_NV12 || convert_src->data[0] == nullptr) {
                std::cerr << "AVFrame error" << std::endl;
                av_frame_unref(frame);
                av_frame_unref(sw_frame);
                av_packet_unref(pkt);
                continue;
            }

            // if (use_hw_accel) { 
            if (false) {
                if (rga_cvt_color(convert_src, outFrame) != 0) {
                    fprintf(stderr, "RGA color conversion failed.\n");
                    // av_frame_unref(frame);
                    // av_frame_unref(sw_frame);
                    // av_packet_unref(pkt);
                    // close();
                }
            } else {
                int src_w = convert_src->width;
                int src_h = convert_src->height;
                enum AVPixelFormat src_fmt = (enum AVPixelFormat)convert_src->format;
                enum AVPixelFormat dst_fmt = AV_PIX_FMT_RGB24;

                if (!sws_ctx) {
                    sws_ctx = sws_getContext(src_w, src_h, src_fmt,
                                             src_w, src_h, dst_fmt,
                                             SWS_BILINEAR, NULL, NULL, NULL);
                    if (!sws_ctx) {
                        close();
                        return AVERROR(EINVAL);
                    }
                }

                int needed = av_image_get_buffer_size(dst_fmt, src_w, src_h, 1);
                if (needed != rgb_bufsize) {
                    av_freep(&rgb_buf);
                    rgb_buf = (uint8_t*)av_malloc(needed);
                    rgb_bufsize = needed;
                }

                uint8_t *dst_data[4] = {0};
                int dst_linesizes[4] = {0};
                av_image_fill_arrays(dst_data, dst_linesizes, rgb_buf, dst_fmt, src_w, src_h, 1);

                sws_scale(sws_ctx,
                          (const uint8_t * const*)convert_src->data,
                          convert_src->linesize,
                          0, src_h,
                          dst_data, dst_linesizes);

                outFrame = cv::Mat(src_h, src_w, CV_8UC3, dst_data[0], dst_linesizes[0]).clone();
            }

            av_frame_unref(frame);
            av_frame_unref(sw_frame);
        }
    }

    av_packet_unref(pkt);
    return 0;
}

void Reader::close() {
    if (rgb_buf) av_freep(&rgb_buf);
    if (sws_ctx) sws_freeContext(sws_ctx);
    av_frame_free(&sw_frame);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec_ctx);
    av_buffer_unref(&hw_device_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
    isOpened = false;
}


#include "server.h"

Server::Server() : oformat_ctx(nullptr), enc_ctx(nullptr), encoder(nullptr), stream(nullptr),
                   frame(nullptr), pkt(nullptr), sws_ctx(nullptr), frame_pts(0), isOpened(false) {}

Server::~Server() {
    close();
}

void Server::print_error(const char *msg, int err) {
    char errbuf[128];
    av_strerror(err, errbuf, sizeof(errbuf));
    fprintf(stderr, "%s: %s\n", msg, errbuf);
}

int Server::open(const std::string& output_url, int width, int height, int fps, bool use_hw) {
    int ret = 0;
    frame_width = width;
    frame_height = height;

    // 1. Allocate format context
    ret = avformat_alloc_output_context2(&oformat_ctx, NULL, "rtsp", output_url.c_str());
    if (ret < 0) {
        print_error("Could not allocate output format context", ret);
        return ret;
    }

    // 2. Find encoder
    if (use_hw) {
        encoder = avcodec_find_encoder_by_name("h264_omx"); // Try OpenMAX encoder first
        if (encoder) {
            fprintf(stdout, "Using hardware encoder: h264_omx\n");
        } else {
            fprintf(stderr, "Hardware encoder h264_omx not found, trying libx264.\n");
        }
    }
    if (!encoder) {
        encoder = avcodec_find_encoder_by_name("libx264");
        if (!encoder) {
            fprintf(stderr, "Encoder libx264 not found\n");
            return AVERROR_ENCODER_NOT_FOUND;
        }
        fprintf(stdout, "Using software encoder: libx264\n");
    }

    // 3. Create new stream
    stream = avformat_new_stream(oformat_ctx, encoder);
    if (!stream) {
        fprintf(stderr, "Failed to create new stream\n");
        return -1;
    }
    stream->id = oformat_ctx->nb_streams - 1;

    // 4. Allocate and configure encoder context
    enc_ctx = avcodec_alloc_context3(encoder);
    if (!enc_ctx) {
        fprintf(stderr, "Failed to allocate encoder context\n");
        return AVERROR(ENOMEM);
    }

    enc_ctx->codec_id = encoder->id;
    enc_ctx->width = frame_width;
    enc_ctx->height = frame_height;
    enc_ctx->time_base = (AVRational){1, fps};
    enc_ctx->framerate = (AVRational){fps, 1};
    enc_ctx->gop_size = 12;
    enc_ctx->max_b_frames = 1;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (encoder->id == AV_CODEC_ID_H264) {
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(enc_ctx->priv_data, "tune", "zerolatency", 0);
    }
    if (oformat_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // 5. Open codec
    ret = avcodec_open2(enc_ctx, encoder, NULL);
    if (ret < 0) {
        print_error("Could not open codec", ret);
        return ret;
    }

    // 6. Copy codec parameters to stream
    ret = avcodec_parameters_from_context(stream->codecpar, enc_ctx);
    if (ret < 0) {
        print_error("Could not copy codec parameters", ret);
        return ret;
    }

    // 7. Open output URL
    if (!(oformat_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&oformat_ctx->pb, output_url.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            print_error("Could not open output URL", ret);
            return ret;
        }
    }

    // 8. Write stream header
    ret = avformat_write_header(oformat_ctx, NULL);
    if (ret < 0) {
        print_error("Error occurred when opening output URL", ret);
        return ret;
    }

    // 9. Allocate frame and packet
    frame = av_frame_alloc();
    if (!frame) return AVERROR(ENOMEM);
    frame->format = enc_ctx->pix_fmt;
    frame->width = enc_ctx->width;
    frame->height = enc_ctx->height;
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) return ret;

    pkt = av_packet_alloc();
    if (!pkt) return AVERROR(ENOMEM);

    isOpened = true;
    fprintf(stdout, "RTSP server opened successfully at: %s\n", output_url.c_str());
    return 0;
}

int Server::encodeFrame(const cv::Mat& inFrame) {
    // std::cout << "send server" << std::endl;
    if (!isOpened) return -1;

    // Ensure input frame has correct dimensions
    if (inFrame.cols != frame_width || inFrame.rows != frame_height) {
        fprintf(stderr, "Input frame dimensions do not match server configuration\n");
        std::cout << inFrame.cols << inFrame.rows << frame_width << frame_height << std::endl;
        return -1;
    }

    // Convert cv::Mat (BGR) to AVFrame (YUV420p)
    if (!sws_ctx) {
        sws_ctx = sws_getContext(frame_width, frame_height, AV_PIX_FMT_RGB24,
                                 frame_width, frame_height, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, NULL, NULL, NULL);
        if (!sws_ctx) return -1;
    }

    const int stride[] = { static_cast<int>(inFrame.step[0]) };
    sws_scale(sws_ctx, &inFrame.data, stride, 0, frame_height, frame->data, frame->linesize);

    frame->pts = frame_pts++;

    // Encode the frame
    int ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        print_error("Error sending a frame for encoding", ret);
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return 0; // Need more data or end of stream
        } else if (ret < 0) {
            print_error("Error during encoding", ret);
            return ret;
        }

        // Rescale timestamps
        av_packet_rescale_ts(pkt, enc_ctx->time_base, stream->time_base);
        pkt->stream_index = stream->index;

        // Write the compressed frame to the media file
        ret = av_interleaved_write_frame(oformat_ctx, pkt);
        if (ret < 0) {
            print_error("Error while writing output packet", ret);
        }
        av_packet_unref(pkt);
    }
    return 0;
}

void Server::close() {
    if (!isOpened) return;

    if (enc_ctx) {
        int ret = encodeFrame(cv::Mat());
        if (ret >= 0) {
            while (ret >= 0) {
                ret = avcodec_receive_packet(enc_ctx, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) {
                    print_error("Error during flushing", ret);
                    break;
                }
                av_interleaved_write_frame(oformat_ctx, pkt);
                av_packet_unref(pkt);
            }
        }
    }

    if (oformat_ctx) av_write_trailer(oformat_ctx);
    if (enc_ctx) avcodec_free_context(&enc_ctx);
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (oformat_ctx && !(oformat_ctx->oformat->flags & AVFMT_NOFILE)) avio_closep(&oformat_ctx->pb);
    if (oformat_ctx) avformat_free_context(oformat_ctx);

    isOpened = false;
    fprintf(stdout, "RTSP server closed.\n");
}

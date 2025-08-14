#include "rga_helper.hpp"

std::mutex rgaMutex;

int resize_rga(const cv::Mat &image, cv::Mat &resized_image) {
    std::lock_guard<std::mutex> lock(rgaMutex);
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    if (image.type() != CV_8UC3) {
        printf("source image type is %d!\n", image.type());
        return -1;
    }

    src = wrapbuffer_virtualaddr((void *)image.data, image.cols, image.rows, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resized_image.data, resized_image.cols, resized_image.rows, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        fprintf(stderr, "rga check error! %s\n", imStrError((IM_STATUS)ret));
        releasebuffer_handle(src.handle);
        releasebuffer_handle(dst.handle);
        return -1;
    }

    IM_STATUS STATUS = imresize(src, dst);

    // release buffer
    releasebuffer_handle(src.handle);
    releasebuffer_handle(dst.handle);
    
    if (STATUS != IM_STATUS_SUCCESS) {
        fprintf(stderr, "imresize error: %s\n", imStrError(STATUS));
        return -1;
    }

    return 0;
}

int rga_cvt_color(AVFrame* src_frame, cv::Mat& dst_mat) {
    std::lock_guard<std::mutex> lock(rgaMutex);

    int ret = 0;
    rga_buffer_t src_img, dst_img;
    rga_buffer_handle_t src_handle, dst_handle;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    int src_w = src_frame->width;
    int src_h = src_frame->height;
    RgaSURF_FORMAT src_fmt;

    switch (src_frame->format) {
        case AV_PIX_FMT_NV12:
            src_fmt = RK_FORMAT_YCbCr_420_SP;
            break;
        case AV_PIX_FMT_YUV420P:
            src_fmt = RK_FORMAT_YCbCr_420_P;
            break;
        default:
            fprintf(stderr, "RGA unsupported source format: %d\n", src_frame->format);
            return -1;
    }

    int dst_w = src_w;
    int dst_h = src_h;
    RgaSURF_FORMAT dst_fmt = RK_FORMAT_RGB_888;

    dst_mat.create(dst_h, dst_w, CV_8UC3);

    src_handle = importbuffer_virtualaddr(src_frame->data[0], src_frame->linesize[0] * src_frame->height * 3 / 2);
    dst_handle = importbuffer_virtualaddr(dst_mat.data, dst_mat.total() * dst_mat.elemSize());

    if (src_handle == 0 || dst_handle == 0) {
        printf("importbuffer failed!\n");
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        return -1;
    }

    src_img = wrapbuffer_handle(src_handle, src_w, src_h, src_fmt);
    dst_img = wrapbuffer_handle(dst_handle, dst_w, dst_h, dst_fmt);

    ret = imcheck(src_img, dst_img, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
        return -1;
    }

    ret = imcvtcolor(src_img, dst_img, src_fmt, dst_fmt);
    if (ret != IM_STATUS_SUCCESS) {
        printf("imcvtcolor failed: %s\n", imStrError((IM_STATUS)ret));
    }

    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);

    return (ret == IM_STATUS_SUCCESS) ? 0 : -1;
}

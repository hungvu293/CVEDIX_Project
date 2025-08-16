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
    int src_width, src_height, src_format;
    int dst_width, dst_height, dst_format;
    char *src_buf, *dst_buf;
    int src_buf_size, dst_buf_size;

    rga_buffer_t src_img, dst_img;
    rga_buffer_handle_t src_handle, dst_handle;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_width = src_frame->width;
    src_height = src_frame->height;
    src_format = RK_FORMAT_YCbCr_420_SP; // Assuming NV12 format

    if (src_width == 0 || src_height == 0) {
        fprintf(stderr, "Invalid source image dimensions!\n");
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        if (src_buf) free(src_buf);
        if (dst_buf) free(dst_buf);
        return -1;
    }

    dst_width = src_width;
    dst_height = src_height;
    dst_format = RK_FORMAT_RGB_888; // Convert to RGB format

    src_buf_size = src_width * src_height * get_bpp_from_format(src_format);
    dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);

    src_buf = (char *)malloc(src_buf_size);
    dst_buf = (char *)malloc(dst_buf_size);

    if (src_frame->format == AV_PIX_FMT_NV12) {
        // Copy Y plane
        for (int i = 0; i < src_height; i++) {
            memcpy(src_buf + i * src_width, src_frame->data[0] + i * src_frame->linesize[0], src_width);
        }
        // Copy UV plane
        for (int i = 0; i < src_height / 2; i++) {
            memcpy(src_buf + src_width * src_height + i * src_width, 
                   src_frame->data[1] + i * src_frame->linesize[1], src_width);
        }
    } else {
        printf("Unsupported AVFrame format!\n");
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        if (src_buf) free(src_buf);
        if (dst_buf) free(dst_buf);
        ret = -1;
    }
    memset(dst_buf, 0x80, dst_buf_size);
    dst_mat.create(dst_height, dst_width, CV_8UC3);

    src_handle = importbuffer_virtualaddr(src_buf, src_buf_size);
    dst_handle = importbuffer_virtualaddr(dst_buf, dst_buf_size);
    if (src_handle == 0 || dst_handle == 0) {
        printf("importbuffer failed!\n");
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        if (src_buf) free(src_buf);
        if (dst_buf) free(dst_buf);
        return -1;
    }

    src_img = wrapbuffer_handle(src_handle, src_width, src_height, src_format);
    dst_img = wrapbuffer_handle(dst_handle, dst_width, dst_height, dst_format);

    ret = imcheck(src_img, dst_img, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        if (src_buf) free(src_buf);
        if (dst_buf) free(dst_buf);
        return -1;
    }

    ret = imcvtcolor(src_img, dst_img, src_format, dst_format);
    if (ret != IM_STATUS_SUCCESS) {
        printf("imcvtcolor failed: %s\n", imStrError((IM_STATUS)ret));
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        if (src_buf) free(src_buf);
        if (dst_buf) free(dst_buf);
        return -1;
    }

    memcpy(dst_mat.data, dst_buf, dst_buf_size);

    if (src_handle) releasebuffer_handle(src_handle);
    if (dst_handle) releasebuffer_handle(dst_handle);
    if (src_buf) free(src_buf);
    if (dst_buf) free(dst_buf);

    return (ret == IM_STATUS_SUCCESS) ? 0 : -1;
}

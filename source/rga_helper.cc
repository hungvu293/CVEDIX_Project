#include "rga_helper.hpp"

std::mutex rgaMutex;

int resize_rga(const cv::Mat &image, cv::Mat &resized_image) {
    // std::lock_guard<std::mutex> lock(rgaMutex);
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
    // std::lock_guard<std::mutex> lock(rgaMutex);
    rga_buffer_t src_img, dst_img;
    im_rect src_rect, dst_rect;
    rga_buffer_handle_t src_handle = 0, dst_handle = 0;
    int ret = 0;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    int src_width = src_frame->width, src_height = src_frame->height;
    if (src_width == 0 || src_height == 0 || src_frame->format != AV_PIX_FMT_NV12) {
        fprintf(stderr, "Invalid source image dimensions or format!\n");
        return -1;
    }
    int src_format = RK_FORMAT_YCbCr_420_SP;
    int src_buf_size = src_width * src_height * get_bpp_from_format(src_format);

    int dst_width = src_width, dst_height = src_height;
    int dst_format = RK_FORMAT_RGB_888;
    dst_mat.create(dst_height, dst_width, CV_8UC3);


    char* src_buf = nullptr;
    if (src_frame->linesize[0] != src_width || src_frame->linesize[1] != src_width) {
        src_buf = (char*)malloc(src_buf_size);
        if (!src_buf) {
            fprintf(stderr, "Failed to allocate source buffer!\n");
            return -1;
        }
        // Copy Y plane
        for (int i = 0; i < src_height; i++) {
            memcpy(src_buf + i * src_width, src_frame->data[0] + i * src_frame->linesize[0], src_width);
        }
        // Copy UV plane
        for (int i = 0; i < src_height / 2; i++) {
            memcpy(src_buf + src_width * src_height + i * src_width,
                   src_frame->data[1] + i * src_frame->linesize[1], src_width);
        }
    }

    // void* src_data = src_buf ? src_buf : (char*)src_frame->data[0];
    void* src_data = (char*)src_frame->data;
    src_img = wrapbuffer_virtualaddr(src_data, src_width, src_height, RK_FORMAT_YCbCr_420_SP);
    dst_img = wrapbuffer_virtualaddr(dst_mat.data, dst_width, dst_height, RK_FORMAT_RGB_888);

    // Set color space properties
    // imsetColorSpace(&src_img, IM_YUV_BT709_LIMIT_RANGE); // NV12 typically uses BT.601
    // imsetColorSpace(&dst_img, IM_RGB_FULL);              // RGB full range

    // Check RGA compatibility
    if (imcheck(src_img, dst_img, src_rect, dst_rect) != IM_STATUS_NOERROR) {
        fprintf(stderr, "rga check error: %s\n", imStrError((IM_STATUS)ret));
        releasebuffer_handle(src_img.handle);
        releasebuffer_handle(dst_img.handle);
        return -1;
    } else {
        // Perform color conversion
        IM_STATUS status = imcvtcolor(src_img, dst_img, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888);
        if (status != IM_STATUS_SUCCESS) {
            fprintf(stderr, "imcvtcolor error: %s\n", imStrError(status));
            releasebuffer_handle(src_img.handle);
            releasebuffer_handle(dst_img.handle);
            return -1;
        }
    }

    releasebuffer_handle(src_img.handle);
    releasebuffer_handle(dst_img.handle);
    if (src_buf)
        free(src_buf);

    return 0;
}

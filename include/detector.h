#ifndef DETECTOR_H
#define DETECTOR_H

#include <cstddef>
#include "rknn_api.h"
#include "im2d.h"
#include "rga.h"
#include "opencv2/opencv.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);
static float clamp(float val, float minv, float maxv);
static float IoU(const cv::Rect& a, const cv::Rect& b);
static void quick_sort_indices(std::vector<float>& scores, std::vector<int>& indices, int left, int right);
static std::vector<int> nms_boxes(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nms_thresh);
static int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size);
static int get_core_num();
struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Detector {
private:
    int ret;
    
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    float nms_threshold {0.5};
    float box_conf_threshold {0.5};
    std::vector<std::string> classes{"plate"};
public:
    Detector(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    std::vector<Detection> infer(cv::Mat &ori_img);
    void draw(cv::Mat &ori_img, const std::vector<Detection> &detections);
    ~Detector();
};

#endif // DETECTOR_H
#ifndef DETECTOR_H
#define DETECTOR_H

#include <cstddef>
#include "rknn_api.h"
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "opencv2/opencv.hpp"


static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);
static float clamp(float val, float minv, float maxv);
static float IoU(const cv::Rect& a, const cv::Rect& b);
static void quick_sort_indices(std::vector<float>& scores, std::vector<int>& indices, int left, int right);
static std::vector<int> nms_boxes(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nms_thresh);
static int get_core_num();

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

struct FrameWithMetadata {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    int camera_id;
    
    FrameWithMetadata() = default;
    FrameWithMetadata(cv::Mat f, std::chrono::system_clock::time_point t, int id) 
        : frame(std::move(f)), capture_time(t), camera_id(id) {}
};

struct DetectionWithMetadata {
    std::vector<Detection> detections;
    std::chrono::system_clock::time_point capture_time;
    int camera_id;
    cv::Mat original_frame;
    
    DetectionWithMetadata() = default;
    DetectionWithMetadata(std::vector<Detection> dets, std::chrono::system_clock::time_point t, 
                         int id, cv::Mat frame) 
        : detections(std::move(dets)), capture_time(t), camera_id(id), original_frame(std::move(frame)) {}
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
    float box_conf_threshold {0.2};
    std::vector<std::string> classes{"plate"};
    // std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

public:
    Detector(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    DetectionWithMetadata infer_meta(FrameWithMetadata input_data) {
        std::vector<Detection> detections = this->infer(input_data.frame); // Call existing infer
        return DetectionWithMetadata(std::move(detections), input_data.capture_time, 
                                   input_data.camera_id, std::move(input_data.frame));
    }
    std::vector<Detection> infer(cv::Mat &ori_img);
    void draw(cv::Mat &ori_img, const std::vector<Detection> &detections);
    int resize_rga(const cv::Mat &src, cv::Mat &dst);
    ~Detector();
};

#endif // DETECTOR_H
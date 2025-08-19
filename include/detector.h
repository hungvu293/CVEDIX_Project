#ifndef DETECTOR8_H
#define DETECTOR8_H

#include <cstddef>
#include "rknn_api.h"
// #include "im2d.h"
// #include "rga.h"
// #include "RgaUtils.h"

#include "rga_helper.hpp"
#include "opencv2/opencv.hpp"

// #include <thread>
#include <mutex>


#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128

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
    DetectionWithMetadata(std::vector<Detection> dets, std::chrono::system_clock::time_point t, int id, cv::Mat frame) 
        : detections(std::move(dets)), capture_time(t), camera_id(id), original_frame(std::move(frame)) {}
};

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
    char name[OBJ_NAME_MAX_SIZE];
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

class Detector {
public:
    Detector(const std::string &model_path);
    ~Detector();
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    DetectionWithMetadata infer_meta(FrameWithMetadata input_data) {
        std::vector<Detection> detections = this->infer(input_data.frame); // Call existing infer
        std::vector<Detection> filtered_detections = this->plateFilter(input_data.frame, detections);
        // this->draw(input_data.frame, detections);
        detections = std::move(filtered_detections);
        return DetectionWithMetadata(std::move(detections), input_data.capture_time,
                                      input_data.camera_id, std::move(input_data.frame));
    }
    std::vector<Detection> infer(cv::Mat &ori_img);
    void draw(cv::Mat &ori_img, const std::vector<Detection> &detections);
    object_detect_result_list*  od_results;
    static std::mutex resize_mutex;

    std::vector<Detection> plateFilter(cv::Mat& ori_img, const std::vector<Detection>& detections);

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
    float box_conf_threshold {0.1};
    std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    // std::vector<std::string> classes{"plate"};

    int postprocess(rknn_output* outputs, float scale_w, float scale_h,
                    float conf_threshold, float nms_threshold);
    
    std::vector<Detection> convert_output(object_detect_result_list* od_results);
};

// static int resize_rga(const cv::Mat &src, cv::Mat &dst);
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int clamp(float val, int min, int max) { 
    return val > min ? (val < max ? val : max) : min;
};
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
static void compute_dfl(float* tensor, int dfl_len, float* box);
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold);
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1);
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold,
                      int classNum);



#endif
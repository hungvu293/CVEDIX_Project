#ifndef YOLOV8_H
#define YOLOV8_H

#include "opencv2/opencv.hpp"
#include "rknn_api.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45f
#define BOX_THRESH 0.1f

static const char* labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
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

class YoloV8 {
public:
    YoloV8();
    ~YoloV8();

    rknn_app_context_t* app_ctx;
    object_detect_result_list*  od_results;

    rknn_input* inputs;
    rknn_output* outputs;
    
    int init(const char* model_path);
    int run(cv::Mat& orig_img);
    int draw(cv::Mat& orig_img);
    int release();

private:
    int read_data_from_file(const char *path, char **out_data);
    void dump_tensor_attr(rknn_tensor_attr* attr);
    int postprocess(rknn_output* outputs, float scale_w, float scale_h,
                    float conf_threshold, float nms_threshold);
};

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
                      float threshold);

#endif
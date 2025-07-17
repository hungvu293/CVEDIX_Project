#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cstdlib>                  // for malloc and free
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "rknn_api.h"

#include <set>
#include <vector>

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.1

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

static void dump_tensor_attr(rknn_tensor_attr* attr);
int read_data_from_file(const char *path, char **out_data);
int init_model(const char* model_path, rknn_app_context_t *app_ctx);
int inference_model(rknn_app_context_t *app_ctx, cv::Mat *orig_img_ptr, object_detect_result_list *od_results);
int postprocess(rknn_app_context_t *app_ctx, void *outputs, float scale_w, float scale_h, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold);
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
static void compute_dfl(float* tensor, int dfl_len, float* box);
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold);
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1);
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);
int release_yolov8_model(rknn_app_context_t *app_ctx);

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("\tindex=%d, name=%s, \n\t\tn_dims=%d, dims=[%d, %d, %d, %d], \n\t\tn_elems=%d, size=%d, fmt=%s, \n\t\ttype=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

int init_model(const char* model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int inference_model(rknn_app_context_t *app_ctx, cv::Mat *orig_img, object_detect_result_list *od_results)
{
    int ret;
    // Init inputs 
    rknn_input inputs[app_ctx->io_num.n_input];
    memset(inputs, 0, sizeof(inputs));

    // Init outputs
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }

    // Preprocess and set inputs
    cv::Mat img;
    cv::cvtColor(*orig_img, img, cv::COLOR_BGR2RGB);
    int img_width  = img.cols;
    int img_height = img.rows;
    // TODO: Use letter box to keep aspect ratio
    if (img_width != app_ctx->model_width || img_height != app_ctx->model_height)
    {
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(app_ctx->model_width, app_ctx->model_height));
        inputs[0].buf = (void*)resized_img.data;
    }
    else
    {
        inputs[0].buf = (void*)img.data;
    }
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get output
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    
    float scale_w = (float)app_ctx->model_width / img_width;
    float scale_h = (float)app_ctx->model_height / img_height;
    const float nms_threshold = NMS_THRESH;
    const float conf_threshold = BOX_THRESH;
    postprocess(app_ctx, outputs, scale_w, scale_h, conf_threshold, nms_threshold, od_results);

    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    return 0;
}

int postprocess(rknn_app_context_t *app_ctx, void *outputs, float scale_w, float scale_h, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
    rknn_output *_outputs = (rknn_output *)outputs;

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    int dfl_len = app_ctx->output_attrs[0].dims[1] /4;

    int output_per_branch = app_ctx->io_num.n_output / 3;

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];

        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     (int8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold);
        }
    }
    
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }
    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        // float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        // float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> score_thres_i8){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}



int release_yolov8_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int main() {
    const char model_path[] = "../model/yolov8.rknn";
    const char img_path[] = "../busstop.jpg";
    
    // Init runtime context
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    cv::Mat orig_img;
    ret = init_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    // Input
    orig_img=cv::imread(img_path, 1);
    if (orig_img.empty())
    {
        printf("Error grabbing img\n");
        goto out;
    }
    
    // Output
    object_detect_result_list od_results;
    memset(&od_results, 0x00, sizeof(object_detect_result_list));
    
    //Inference
    ret = inference_model(&rknn_app_ctx, &orig_img, &od_results);
    if (ret != 0)
    {
        printf("inference fail\n");
        goto out;
    }

    char text[256];

    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%d @ (%d %d %d %d) %.3f\n", det_result->cls_id,
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2),cv::Scalar(255, 0, 0));
    }
    
    // imshow("Radxa zero 3W - 1,8 GHz - 4 Mb RAM", orig_img);
    // char esc = cv::waitKey(5);
    // if(esc == 27) break;
    imwrite("../out.jpg", orig_img);
out:
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }
    return 0;
}
#include <vector>
#include "detector.h"
#include <mutex>
const int RK3566 = 1;

static int get_core_num()
{
    static int core_num = 0;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);

    int temp = core_num % RK3566;
    core_num++;
    return temp;
}

int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size) {
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    size_t img_width = image.cols;
    size_t img_height = image.rows;
    if (image.type() != CV_8UC3)
    {
        printf("source image type is %d!\n", image.type());
        return -1;
    }
    size_t target_width = target_size.width;
    size_t target_height = target_size.height;
    src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
        return -1;
    }
    IM_STATUS STATUS = imresize(src, dst);
    return 0;
}

static float clamp(float val, float minv, float maxv) {
    if (val < minv) return minv;
    if (val > maxv) return maxv;
    return val;
}

// Intersection over Union
static float IoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? (float)interArea / unionArea : 0.f;
}

// Quick sort indices by score (descending)
static void quick_sort_indices(std::vector<float>& scores, std::vector<int>& indices, int left, int right) {
    if (left >= right) return;
    float pivot = scores[indices[left]];
    int l = left, r = right;
    while (l < r) {
        while (l < r && scores[indices[r]] <= pivot) r--;
        while (l < r && scores[indices[l]] >= pivot) l++;
        if (l < r) std::swap(indices[l], indices[r]);
    }
    std::swap(indices[left], indices[l]);
    quick_sort_indices(scores, indices, left, l - 1);
    quick_sort_indices(scores, indices, l + 1, right);
}

// Manual NMS
static std::vector<int> nms_boxes(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nms_thresh) {
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) indices[i] = i;
    quick_sort_indices(const_cast<std::vector<float>&>(scores), indices, 0, (int)indices.size() - 1);
    std::vector<int> keep;
    std::vector<bool> removed(boxes.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (removed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            if (removed[idx2]) continue;
            if (IoU(boxes[idx], boxes[idx2]) > nms_thresh) {
                removed[idx2] = true;
            }
        }
    }
    return keep;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    // printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
    //        "type=%s, qnt_type=%s, "
    //        "zp=%d, scale=%f\n",
    //        attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
    //        attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
    //        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

Detector::Detector(const std::string &model_path) : model_path(model_path) {
    this->model_path = model_path;
}

Detector::~Detector() {
    ret = rknn_destroy(ctx);

    if (model_data)
        free(model_data);

    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
}

int Detector::init(rknn_context *ctx_in, bool share_weight) {
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    if (model_data == NULL) {
        printf("load model %s failed!\n", model_path.c_str());
        return -1;
    }
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // rknn_core_mask core_mask;
    // switch (get_core_num())
    // {
    // case 0:
    //     core_mask = RKNN_NPU_CORE_0;
    //     break;
    // case 1:
    //     core_mask = RKNN_NPU_CORE_1;
    //     break;
    // case 2:
    //     core_mask = RKNN_NPU_CORE_2;
    //     break;
    // }
    // ret = rknn_set_core_mask(ctx, core_mask);
    // if (ret < 0)
    // {
    //     printf("rknn_init core error ret=%d\n", ret);
    //     return -1;
    // }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入参数/Set the input parameters
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 设置输出参数/Set the output parameters
    // output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    // for (int i = 0; i < io_num.n_output; i++)
    // {
    //     output_attrs[i].index = i;
    //     ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    //     dump_tensor_attr(&(output_attrs[i]));
    // }
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error! rknn_query fail! ret=%d\n", ret);
            return -1;
        }

        printf("Output tensor %d details:\n", i);
        printf("  index: %d\n", output_attrs[i].index);
        printf("  name: %s\n", output_attrs[i].name);
        printf("  n_dims: %d\n", output_attrs[i].n_dims);
        
        // In ra kích thước của từng chiều
        printf("  dims: [");
        for (int j = 0; j < output_attrs[i].n_dims; ++j) {
            printf("%d", output_attrs[i].dims[j]);
            if (j < output_attrs[i].n_dims - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("  n_elems: %d\n", output_attrs[i].n_elems);
        printf("  size: %d\n", output_attrs[i].size);
        printf("  fmt: %d (RKNN_TENSOR_NCHW=0, RKNN_TENSOR_NHWC=1)\n", output_attrs[i].fmt);
        printf("  type: %d (RKNN_TENSOR_UINT8=0, RKNN_TENSOR_FLOAT16=1, RKNN_TENSOR_INT8=2, ...)\n", output_attrs[i].type);
        printf("  scale: %f\n", output_attrs[i].scale);
        printf("  zp: %d\n", output_attrs[i].zp);
    }
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}

rknn_context *Detector::get_pctx()
{
    return &ctx;
}

std::vector<Detection> Detector::infer(cv::Mat &ori_img) {
    img_width = ori_img.cols;
    img_height = ori_img.rows;

    cv::Size target_size(width, height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);

    float scale_w = (float)target_size.width / img_width;
    float scale_h = (float)target_size.height / img_height;

    if (img_width != width || img_height != height) {
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        ret = resize_rga(src, dst, ori_img, resized_img, target_size);
        if (ret != 0) {
            std::cerr << "resize rga error" << std::endl;
        }
        inputs[0].buf = resized_img.data;
    }
    else {
        inputs[0].buf = ori_img.data;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 1;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int num_anchors = 8400;
    int num_values_per_anchor = 5; 
    float* output_data = (float*)outputs[0].buf;

    float* cx_data = output_data + 0 * num_anchors;
    float* cy_data = output_data + 1 * num_anchors;
    float* w_data = output_data + 2 * num_anchors;
    float* h_data = output_data + 3 * num_anchors;
    float* conf_data = output_data + 4 * num_anchors;

    for (int i = 0; i < num_anchors; ++i) {
        float confidence = conf_data[i];

        if (confidence > box_conf_threshold) {
            float cx = cx_data[i];
            float cy = cy_data[i];
            float w  = w_data[i];
            float h  = h_data[i];

            int x1 = static_cast<int>((cx - 0.5f * w) / scale_w);
            int y1 = static_cast<int>((cy - 0.5f * h) / scale_h);
            int box_w = static_cast<int>(w / scale_w);
            int box_h = static_cast<int>(h / scale_h);

            boxes.push_back(cv::Rect(x1, y1, box_w, box_h));
            confidences.push_back(confidence);
            class_ids.push_back(0);
        }
    }

    float nms_thresh = nms_threshold;
    std::vector<int> keep = nms_boxes(boxes, confidences, nms_thresh);
    for (size_t i = 0; i < keep.size(); ++i) {
        int idx = keep[i];
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        if (det.class_id >= 0 && det.class_id < (int)classes.size())
            det.className = classes[det.class_id];
        else
            det.className = "";
        det.color = cv::Scalar(0, 255, 0);
        detections.push_back(det);
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    return detections;
}

void Detector::draw(cv::Mat &ori_img, const std::vector<Detection> &detections) {
    for (const auto& det : detections) {
        cv::rectangle(ori_img, det.box, det.color, 2);
        std::string label = det.className + ": " + std::to_string(det.confidence);
        cv::putText(ori_img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, det.color, 2);
        std::cout << "size: " << detections.size() << std::endl;
        std::cout << "box" << det.box << " "
                  << "confidence: " << det.confidence << " "
                  << "class_id: " << det.class_id << std::endl;
    }
}


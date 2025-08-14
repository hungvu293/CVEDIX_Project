#include "detector.h"

#include <set>
#include <vector>
#include <string.h>

#include <chrono>

std::mutex Detector::resize_mutex;

Detector::Detector(const std::string &model_path) : model_path(model_path) {
    this->model_path = model_path;
    od_results = new object_detect_result_list();
}
Detector::~Detector() {
    ret = rknn_destroy(ctx);

    if (model_data)
        free(model_data);
    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
    if (od_results) {
        delete od_results;
        od_results = nullptr;
    }
}

int Detector::init(rknn_context *ctx_in, bool share_weight)
{
    // printf("[LOG] Detector::init - Start initializing model\n");
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
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }
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
    // inputs[0].type = RKNN_TENSOR_INT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    printf("[LOG] Detector::init - Initialization finished\n");
    return 0;
}

rknn_context *Detector::get_pctx()
{
    return &ctx;
}


std::vector<Detection> Detector::infer(cv::Mat &ori_img)
{
    
    // printf("[LOG] Detector::infer - Start inference\n");
    img_width = ori_img.cols;
    img_height = ori_img.rows;
    auto start = std::chrono::steady_clock::now();
    cv::Mat resized_img(height, width, CV_8UC3);

    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height;
    
    // detector_mutex.lock();
    if (img_width != width || img_height != height) {
        // ret = resize_rga(ori_img, resized_img);
        // if (ret != 0) {
        //     std::cerr << "resize rga error" << std::endl;
        // }
        cv::resize(ori_img, resized_img, cv::Size(width, height), 0, 0, cv::INTER_LINEAR); 
    }
    else {
        resized_img = ori_img;
    }
    inputs[0].buf = resized_img.data;

    // cv::imwrite("../rga_img.jpg", resized_img);
    // std::cout << "done write resize img" << std::endl;

    // detector_mutex.unlock();
    auto end_resize = std::chrono::steady_clock::now();

    // printf("[LOG] Detector::infer - Input image ready\n");

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    // printf("[LOG] Detector::infer - Running model\n");
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    auto end_run = std::chrono::steady_clock::now();

    // printf("[LOG] Detector::infer - Postprocessing outputs\n");
    postprocess(outputs, scale_w, scale_h, box_conf_threshold, nms_threshold);

    std::vector<Detection> detections;
    if (od_results != nullptr) {
        detections = convert_output(od_results);
    } else {
        detections.clear();
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    auto end_post = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> resize_duration = end_resize - start;
    std::chrono::duration<double, std::milli> run_duration = end_run - end_resize;
    std::chrono::duration<double, std::milli> post_duration = end_post - end_run;
    // std::cout << "resize: " << resize_duration.count() << " ms" << std::endl;
    // std::cout << "run: " << run_duration.count() << " ms" << std::endl;
    // std::cout << "post: " << post_duration.count() << " ms" << std::endl;
    // printf("[LOG] Detector::infer - Inference finished, detections count: %zu\n", detections.size());
    return detections;
}

int Detector::postprocess(rknn_output* outputs, float scale_w, float scale_h, float conf_threshold, float nms_threshold)
{
    // printf("[LOG] Detector::postprocess - Start postprocessing\n");
    if (od_results != nullptr) {
        memset(od_results, 0, sizeof(object_detect_result_list));
    }
    rknn_output *_outputs = (rknn_output *)outputs;

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = width;
    int model_in_h = height;

    int dfl_len = output_attrs[0].dims[1] /4;

    int output_per_branch = io_num.n_output / 3;

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

        grid_h = output_attrs[box_idx].dims[2];
        grid_w = output_attrs[box_idx].dims[3];

        stride = model_in_h / grid_h;

        validCount += process_i8((int8_t *)_outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
                                    (int8_t *)_outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
                                    (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                    grid_h, grid_w, stride, dfl_len, 
                                    filterBoxes, objProbs, classId, conf_threshold, classes.size());
        // printf("[LOG] Detector::postprocess - Branch %d processed, validCount: %d\n", i, validCount);
    }
    
    if (validCount <= 0)
    {
        // printf("[LOG] Detector::postprocess - No valid detections found\n");
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
    std::cout << "detections number: " << od_results->count << std::endl;
    // printf("[LOG] Detector::postprocess - Postprocessing finished, final count: %d\n", last_count);
    return 0;
}

void Detector::draw(cv::Mat &ori_img, const std::vector<Detection> &detections) {
    // printf("[LOG] Detector::draw - Drawing detections\n");
    for (const auto& det : detections) {
        cv::rectangle(ori_img, det.box, det.color, 2);
        std::string label = det.className + ": " + std::to_string(det.confidence);
        cv::putText(ori_img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, det.color, 2);
        std::cout << "size: " << detections.size() << std::endl;
        std::cout << "box" << det.box << " "
                  << "confidence: " << det.confidence << " "
                  << "class_id: " << det.class_id << std::endl;
    }
    // printf("[LOG] Detector::draw - Drawing finished\n");
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold,
                      int classNum)
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
            for (int c= 0; c< classNum; c++){
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

// static int resize_rga(const cv::Mat &image, cv::Mat &resized_image) {
//     rga_buffer_t src_img, dst_img;
//     int src_width = image.rows;
//     int src_height = image.cols;
//     int src_format = RK_FORMAT_RGB_888;

//     int dst_width = resized_image.rows;
//     int dst_height = resized_image.cols;
//     int dst_format = RK_FORMAT_RGB_888;
    
//     int src_buf_size = src_width * src_height * get_bpp_from_format(src_format);
//     int dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);

//     char* src_buf = nullptr;
//     char* dst_buf = nullptr;
//     src_buf = (char*)malloc(src_buf_size);
//     dst_buf = (char*)malloc(dst_buf_size);

//     memcpy(src_buf, image.data, src_buf_size);

//     rga_buffer_handle_t src_handle = 0;
//     rga_buffer_handle_t dst_handle = 0;
//     src_handle = importbuffer_virtualaddr(src_buf, src_buf_size);
//     dst_handle = importbuffer_virtualaddr(dst_buf, dst_buf_size);

//     src_img = wrapbuffer_handle(src_handle, src_width, src_height, src_format);
//     dst_img = wrapbuffer_handle(dst_handle, dst_width, dst_height, dst_format);

//     int ret = imresize(src_img, dst_img, (float)dst_width / src_width, (float)dst_height / src_height);
//     if (ret != IM_STATUS_SUCCESS) {
//         std::cerr << "RGA resize error: " << imStrError((IM_STATUS)ret) << "\n";
//         if (src_handle) releasebuffer_handle(src_handle);
//         if (dst_handle) releasebuffer_handle(dst_handle);
//         if (src_buf) free(src_buf);
//         if (dst_buf) free(dst_buf);
//         return -1;
//     }
//     memcpy(resized_image.data, dst_buf, dst_buf_size);
//     if (src_handle) releasebuffer_handle(src_handle);
//     if (dst_handle) releasebuffer_handle(dst_handle);
//     if (src_buf) free(src_buf);
//     if (dst_buf) free(dst_buf);
//     return 0;
// }




std::vector<Detection> Detector::convert_output(object_detect_result_list* od_results) {
    // printf("[LOG] Detector::convert_output - Start converting output\n");
    std::vector<Detection> output;
    if (od_results == nullptr) {
        // printf("[LOG] Detector::convert_output - od_results is nullptr\n");
        return output;
    }
    for (int i = 0; i < od_results->count; i++) {
        Detection detection;
        detection.class_id = od_results->results[i].cls_id;
        detection.className = classes[detection.class_id];
        detection.confidence = od_results->results[i].prop;
        detection.box = cv::Rect(
            od_results->results[i].box.left,
            od_results->results[i].box.top,
            od_results->results[i].box.right - od_results->results[i].box.left,
            od_results->results[i].box.bottom - od_results->results[i].box.top
        );
        detection.color = cv::Scalar(0, 255, 0);
        output.push_back(detection);
    }
    // printf("[LOG] Detector::convert_output - Conversion finished, output size: %zu\n", output.size());
    return output;
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
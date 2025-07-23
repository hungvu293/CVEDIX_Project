#include "track.h"

/**
@brief Convert Vector to Matrix
@param data
@return Eigen::Matrix<float, Eigen::Dynamic, 6>
*/
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

Tracking::Tracking(float det_thresh_, int max_age_, int min_hits_, float iou_threshold_,
                   int delta_t_, std::string asso_func_, float inertia_, bool use_byte_)
    : oc_sort_tracker(det_thresh_, max_age_, min_hits_, iou_threshold_,
                      delta_t_, asso_func_, inertia_, use_byte_) {}
Tracking::~Tracking() {}

std::vector<Detection> Tracking::convert_output(object_detect_result_list* od_results) {
    std::vector<Detection> output;
    if (od_results == nullptr) {
        return output;
    }
    for (int i = 0; i < od_results->count; i++) {
        Detection detection;
        detection.class_id = od_results->results[i].cls_id;
        detection.className = std::string(od_results->results[i].name);
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
    return output;
}

void Tracking::run(cv::Mat& frame, std::vector<Detection>& output) {
    std::vector<std::vector<float>> data;
    cv::Rect box;
    for (int i = 0; i < output.size(); ++i) {
        Detection detection = output[i];
        box = detection.box;
        std::vector<float> row;

        row.push_back(box.x);
        row.push_back(box.y);
        row.push_back(box.x + box.width);
        row.push_back(box.y + box.height);
        row.push_back(detection.confidence);
        row.push_back(detection.class_id);

        data.push_back(row);

        if (!data.empty()) {
            std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(Vector2Matrix(data));

            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            std::string classString = detection.className + '(' + std::to_string(detection.confidence).substr(0, 4) + ')';
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y + box.height - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 0);

            for (auto j : res) {
                int ID = int(j[4]);
                int Class = int(j[5]);
                float conf = j[6];
                cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), cv::Scalar(0, 0, 255), 1);
            }

            data.clear();
        }
    }
}

// int draw_tracks(cv::Mat& frame, std::vector<Eigen::RowVectorXf>& res) {}

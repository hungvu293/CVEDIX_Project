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
Tracking::~Tracking() {
    // oc_sort_tracker.clear();
    std::cout << "Tracking object destroyed." << std::endl;
}

std::vector<Eigen::RowVectorXf> Tracking::run(cv::Mat& frame, std::vector<Detection>& output) {
    if (output.empty()) {
        return std::vector<Eigen::RowVectorXf>();
    }
    
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
    }
    std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(Vector2Matrix(data));
    data.clear();

    return res;
}

void Tracking::draw_tracks(cv::Mat& frame, std::vector<Eigen::RowVectorXf>& res) {
    if (res.empty())
        return;
    for (auto j : res) {
        int ID = int(j[4]);
        int Class = int(j[5]);
        float conf = j[6];
        cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), cv::Scalar(j(7), j(8), j(9)), 1);
    }
}

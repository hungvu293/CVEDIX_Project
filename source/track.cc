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
    std::cout << "Tracking object destroyed." << std::endl;
}

void Tracking::run(cv::Mat& frame, std::vector<Detection>& output) {
    if (output.empty()) {
        Eigen::Matrix<float, Eigen::Dynamic, 6> dets;
        std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(dets);
        return;
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
        if (!data.empty()) {
            std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(Vector2Matrix(data));
            data.clear();
        }
    }

    return;
}

void Tracking::draw_tracks(cv::Mat& frame) {
    for (int i = 0; i < oc_sort_tracker.trackers.size(); i++) {
        Eigen::Matrix<float, 1, 4> d;
        d = oc_sort_tracker.trackers.at(i).get_state();
        cv::Rect box(d(0), d(1), d(2) - d(0) + 1, d(3) - d(1) + 1);
        cv::Scalar color(oc_sort_tracker.trackers.at(i).color[0], oc_sort_tracker.trackers.at(i).color[1], oc_sort_tracker.trackers.at(i).color[2]);
        cv::putText(frame, cv::format("ID:%d, Last update: %d", oc_sort_tracker.trackers.at(i).id, oc_sort_tracker.trackers.at(i).time_since_update), cv::Point(d(0), d(1) - 5), 0, 0.5, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
        cv::rectangle(frame, box, color, 3);
    }
}

std::vector<cv::Rect> Tracking::getPlates() {
    std::vector<cv::Rect> res;
    float scale_w = 2.0f;
    float scale_h = 8.0f;
    int padding = 25;
    for (int i = 0; i < oc_sort_tracker.trackers.size(); i++) {
        if (!oc_sort_tracker.trackers.at(i).isSent) {
            Eigen::Matrix<float, 1, 4> d;
            d = oc_sort_tracker.trackers.at(i).get_state();
            int old_w = d(2) - d(0) + 1;
            int old_h = d(3) - d(1) + 1;
            int new_w = old_w / scale_w + padding;
            int new_h = old_h / scale_h + padding;
            int x = d(0) + (old_w - new_w) / 2;
            int y = d(1) + (old_h - new_h) / 2;
            cv::Rect box(x, y, new_w, new_h);
            res.push_back(box);
            oc_sort_tracker.trackers.at(i).isSent = true;
        }
    }
    return res;
}
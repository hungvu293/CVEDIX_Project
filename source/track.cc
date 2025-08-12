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

void Tracking::run(cv::Mat& frame, std::vector<Detection>& output) {
    // if (output.empty()) {
    //     if (oc_sort_tracker.trackers.size() != 0) {
    //         for (int i = 0; i < oc_sort_tracker.trackers.size(); i++) {
    //             auto pos = oc_sort_tracker.trackers.at(i).predict();
    //             oc_sort_tracker.trackers.at(i).update(nullptr, 0);
    //             std::cout << "update at " << pos.transpose() << std::endl;
    //         }
    //     }
    //     return;
    // }
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
    }
    std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(Vector2Matrix(data));

    return;
}

void Tracking::draw_tracks(cv::Mat& frame) {
    std::cout << "track size" << oc_sort_tracker.trackers.size() << std::endl;
    for (int i = 0; i < oc_sort_tracker.trackers.size(); i++) {
        Eigen::Matrix<float, 1, 4> d;
        d = oc_sort_tracker.trackers.at(i).get_state();
        cv::Rect box(d(0), d(1), d(2) - d(0) + 1, d(3) - d(1) + 1);
        cv::Scalar color(oc_sort_tracker.trackers.at(i).color[0], oc_sort_tracker.trackers.at(i).color[1], oc_sort_tracker.trackers.at(i).color[2]);
        cv::putText(frame, cv::format("ID:%d, age: %d", oc_sort_tracker.trackers.at(i).id, oc_sort_tracker.trackers.at(i).age), cv::Point(d(0), d(1) - 5), 0, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::rectangle(frame, box, color, 2);
    }
}

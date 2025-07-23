#ifndef TRACKING_H
#define TRACKING_H

#include "opencv2/opencv.hpp"

#include "OCSort.hpp"
#include "inference.h"
#include "yolov8.h"

#include <vector>
#include <string>
#include <Eigen/Dense>

class Tracking {
public:
    Tracking(float det_thresh_ = 0.5f, int max_age_ = 30, int min_hits_ = 3, float iou_threshold_ = 0.3f,
             int delta_t_ = 3, std::string asso_func_ = "iou", float inertia_ = 0.2f, bool use_byte_ = false);
    ~Tracking();
    std::vector<Detection> convert_output(object_detect_result_list* od_results);
    void run(cv::Mat& frame, std::vector<Detection>& output);
    // int draw_tracks(cv::Mat& frame, std::vector<Detection>& output);
    // int draw_detections(cv::Mat& frame, std::vector<Eigen::RowVectorXf>& res);
private:
    ocsort::OCSort oc_sort_tracker;
};

#endif
#ifndef TRACKING_H
#define TRACKING_H

#include "opencv2/opencv.hpp"

#include "OCSort.hpp"
#include "detector.h"

#include <vector>
#include <string>
#include <Eigen/Dense>

class Tracking {
public:
    Tracking(float det_thresh_ = 0.3f, int max_age_ = 15, int min_hits_ = 1, float iou_threshold_ = 0.1f,
             int delta_t_ = 3, std::string asso_func_ = "iou", float inertia_ = 0.2f, bool use_byte_ = false);
    ~Tracking();
    // std::vector<Detection> convert_output(object_detect_result_list* od_results);
    void run(cv::Mat& frame, std::vector<Detection>& output);
    void draw_tracks(cv::Mat& frame);
private:
    ocsort::OCSort oc_sort_tracker;
};

#endif
#include "osd.h"

OSD::OSD() {};

void OSD::show(cv::Mat& frame) {
    // cv::Mat bgr_frame;
    // cv::cvtColor(frame, bgr_frame, cv::COLOR_RGB2BGR);
    // cv::imshow("OSD", bgr_frame);
    // cv::Mat resized_frame;
    // cv::resize(frame, resized_frame, cv::Size(640, 480));
    // cv::imshow("OSD", resized_frame);
    cv::imshow("OSD", frame);
}

void OSD::release() {
    cv::destroyAllWindows();
}

OSD::~OSD() {
    release();
}
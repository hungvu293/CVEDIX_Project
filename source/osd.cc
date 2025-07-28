#include "osd.h"

OSD::OSD() {};

void OSD::show(cv::Mat& frame) {
    cv::Mat bgr_frame;
    cv::cvtColor(frame, bgr_frame, cv::COLOR_RGB2BGR);
    cv::imshow("OSD", bgr_frame);
}

void OSD::release() {
    cv::destroyAllWindows();
}

OSD::~OSD() {
    release();
}
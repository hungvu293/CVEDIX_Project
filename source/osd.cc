#include <osd.h>

OSD::OSD() {};

void OSD::show(cv::Mat& frame) {
    cv::imshow("OSD", frame);
}

void OSD::release() {
    cv::destroyAllWindows();
}

OSD::~OSD() {
    release();
}
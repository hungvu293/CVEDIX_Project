#include <opencv2/highgui.hpp>
#include <stdio.h>
#include "reader.h"

Reader::Reader() {}

int Reader::init(const char* source) {
    cap.open(source);
    if (!cap.isOpened()) {
        printf("source init failed!");
        isInitialized = false;
        return -1;
    }
    isInitialized = true;
    return 0;
}

int Reader::read(cv::Mat& frame) {
    if (!cap.isOpened()) {
        return -1;
    }
    cap.read(frame);
    return 0;
}

void Reader::release() {
    cap.release();
}

Reader::~Reader() {
    release();
}
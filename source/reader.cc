#include <opencv2/highgui.hpp>
#include <stdio.h>
#include "reader.h"
#include <string>

Reader::Reader() {}

int Reader::init(const std::string& source) {
    std::string gstreamer_pipeline_string;
    if (source.rfind("rtsp://",0) == 0) {
    gstreamer_pipeline_string = "rtspsrc location=" + source + " ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1";
    }
    else if (source.rfind("/dev/video",0) == 0) {
        gstreamer_pipeline_string = "v4l2src device=" + source + " ! "
                                    "video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! "
                                    "videorate ! video/x-raw,framerate=15/1 ! "
                                    "videoconvert ! appsink";
    }
    // cap.open(gstreamer_pipeline_string, cv::CAP_GSTREAMER);
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
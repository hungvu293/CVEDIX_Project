#ifndef OSD_H
#define OSD_H
#include <opencv2/opencv.hpp>

class OSD {
public:
    OSD();
    void show(cv::Mat& frame);
    void release();
    ~OSD();
};

#endif // THREADSAFE_QUEUE_H
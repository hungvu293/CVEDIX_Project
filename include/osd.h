#ifndef OSD_H
#define OSD_H
#include <opencv2/opencv.hpp>

class OSD {
public:
    OSD();
    ~OSD();
    cv::Mat combineFrames(const cv::Mat& frame1, const cv::Mat& frame2);
    void init_display();
    void show(cv::Mat& frame);
    void show2(cv::Mat& frame1, cv::Mat& frame2);
    void release();
    std::chrono::steady_clock::time_point last_update_time;
    double fps = 0.0;
};

#endif // OSD_H
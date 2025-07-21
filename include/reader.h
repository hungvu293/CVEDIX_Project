#ifndef READER_H
#define READER_H
#include <opencv2/opencv.hpp>
#include <stdio.h>

class Reader {
public:
    bool isInitialized;
    Reader();
    int init(const char* source);
    int read(cv::Mat& frame);
    void release();
    ~Reader();
private:
    cv::VideoCapture cap;
};

#endif // READER_H
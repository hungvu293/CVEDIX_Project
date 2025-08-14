#ifndef STREAM_H
#define STREAM_H

#include <opencv2/opencv.hpp>
#include <crow.h>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>

class Stream {
public:
    Stream();
    ~Stream();

    void startWebServer(int port = 8080);
    void stopWebServer();
    void updateFrames(const cv::Mat& frame1, const cv::Mat& frame2);
    void updateFrame(const cv::Mat& frame, int camera_id);

private:
    cv::Mat combineFrames(const cv::Mat& frame1, const cv::Mat& frame2);

    std::unique_ptr<crow::SimpleApp> app;
    std::unique_ptr<std::thread> server_thread;
    bool server_running = false;
    
    std::mutex frame_mutex; // For combined frame
    std::vector<uchar> jpeg_buffer;
    bool frame_ready = false;

    // For individual streams
    std::vector<std::mutex> individual_frame_mutexes;
    std::vector<std::vector<uchar>> individual_jpeg_buffers;
    std::vector<bool> individual_frames_ready;

    // For FPS calculation
    std::chrono::steady_clock::time_point last_update_time;
    double fps = 0.0;
    std::vector<std::chrono::steady_clock::time_point> individual_last_update_times;
    std::vector<double> individual_fps;
    static const int NUM_STREAMS = 2;
};

#endif // STREAM_H

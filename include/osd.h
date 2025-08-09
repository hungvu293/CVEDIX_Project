#ifndef OSD_H
#define OSD_H
#include <opencv2/opencv.hpp>
#include <crow.h>
#include <mutex>
#include <thread>
#include <vector>
#include <array>

class OSD {
public:
    OSD();
    void init_display();
    void show(cv::Mat& frame);
    void release();
    ~OSD();
    
    // Web streaming functionality
    void startWebServer(int port = 8080);
    void stopWebServer();
    void updateFrames(const std::array<cv::Mat, 2>& frames, const std::array<bool, 2>& camera_status);

private:
    // Web server components
    std::unique_ptr<crow::SimpleApp> app;
    std::unique_ptr<std::thread> server_thread;
    bool server_running = false;
    
    // Frame data for streaming
    std::mutex frame_mutex;
    std::vector<uchar> combined_jpeg_buffer;
    bool frame_ready = false;
    
    // Helper methods
    cv::Mat combineFrames(const std::array<cv::Mat, 2>& frames, const std::array<bool, 2>& camera_status);
};

#endif // OSD_H
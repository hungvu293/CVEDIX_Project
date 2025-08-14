#include "osd.h"
#include <iostream>

OSD::OSD() {
    // Constructor is now empty
}

OSD::~OSD() {
    release();
}

void OSD::init_display() {
    // Implementation for init_display if needed
}

void OSD::show(cv::Mat& frame) {
    cv::imshow("OSD", frame);
}

void OSD::release() {
    cv::destroyAllWindows();
}


cv::Mat OSD::combineFrames(const cv::Mat& frame1_in, const cv::Mat& frame2_in) {
    
    cv::Mat combined;
    cv::Mat frame1, frame2;
    
    // Chuẩn bị frame 1
    if (!frame1_in.empty()) {
        cv::resize(frame1_in, frame1, cv::Size(640, 480));
    } else {
        frame1 = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame1, "Camera 0: Offline", cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }
    
    // Chuẩn bị frame 2
    if (!frame2_in.empty()) {
        cv::resize(frame2_in, frame2, cv::Size(640, 480));
    } else {
        frame2 = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame2, "Camera 1: Offline", cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }
    
    // Kết hợp hai frame theo chiều ngang
    cv::hconcat(frame1, frame2, combined);
    
    // // Thêm timestamp
    // auto now = std::chrono::system_clock::now();
    // auto time_t = std::chrono::system_clock::to_time_t(now);
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
    // cv::putText(combined, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Add FPS
    std::stringstream fps_ss;
    fps_ss << "FPS: " << std::fixed << std::setprecision(2) << fps;
    cv::putText(combined, fps_ss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    return combined;
}

void OSD::show2(cv::Mat& frame1, cv::Mat& frame2) {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_time);
    last_update_time = now;
    if (duration.count() > 0) {
        double current_fps = 1000.0 / duration.count();
        // fps = (fps * 0.9) + (current_fps * 0.1); // Simple moving average
        fps = current_fps;
    }

    cv::Mat combined = combineFrames(frame1, frame2);
    cv::imshow("OSD", combined);
}


#include "stream.h"
#include <iostream>
#include <chrono>
#include <iomanip>

Stream::Stream() : individual_frame_mutexes(NUM_STREAMS), 
                   individual_jpeg_buffers(NUM_STREAMS), 
                   individual_frames_ready(NUM_STREAMS, false),
                   individual_last_update_times(NUM_STREAMS),
                   individual_fps(NUM_STREAMS, 0.0) {
    app = std::make_unique<crow::SimpleApp>();
    crow::logger::setLogLevel(crow::LogLevel::Warning);
    last_update_time = std::chrono::steady_clock::now();
}

Stream::~Stream() {
    stopWebServer();
}

void Stream::startWebServer(int port) {
    if (server_running) {
        std::cout << "Web server already running" << std::endl;
        return;
    }

    CROW_ROUTE((*app), "/stream")
    ([this](const crow::request& req) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (!frame_ready || jpeg_buffer.empty()) {
            return crow::response(404, "No frame available");
        }

        crow::response res;
        res.set_header("Content-Type", "image/jpeg");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Access-Control-Allow-Origin", "*");
        res.body = std::string(jpeg_buffer.begin(), jpeg_buffer.end());
        return res;
    });

    CROW_ROUTE((*app), "/stream/<int>")
    ([this](int camera_id) {
        if (camera_id < 0 || camera_id >= NUM_STREAMS) {
            return crow::response(404, "Invalid camera ID");
        }

        std::lock_guard<std::mutex> lock(individual_frame_mutexes[camera_id]);
        if (!individual_frames_ready[camera_id] || individual_jpeg_buffers[camera_id].empty()) {
            return crow::response(404, "No frame available for this camera");
        }

        crow::response res;
        res.set_header("Content-Type", "image/jpeg");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Access-Control-Allow-Origin", "*");
        res.body = std::string(individual_jpeg_buffers[camera_id].begin(), individual_jpeg_buffers[camera_id].end());
        return res;
    });

    CROW_ROUTE((*app), "/")
    ([](const crow::request& req) {
        std::string html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Camera Stream</title>
    <style>
        body { margin: 0; padding: 20px; background-color: #f0f0f0; font-family: Arial, sans-serif; text-align: center; }
        .container { display: flex; justify-content: center; align-items: flex-start; flex-wrap: wrap; gap: 20px; }
        .stream-box { border: 2px solid #333; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); padding: 10px; background-color: #fff; }
        img { max-width: 100%; height: auto; display: block; }
        h1 { color: #333; margin-bottom: 30px; width: 100%; }
        h2 { color: #555; margin-top: 0; }
    </style>
</head>
<body>
    <h1>Multi-Camera Stream</h1>
    <div class="container">
        <div class="stream-box">
            <h2>Camera 0</h2>
            <img id="stream0" src="/stream/0" alt="Camera 0 Stream">
        </div>
        <div class="stream-box">
            <h2>Camera 1</h2>
            <img id="stream1" src="/stream/1" alt="Camera 1 Stream">
        </div>
    </div>
    <script>
        function refreshImage(id) {
            const img = document.getElementById('stream' + id);
            img.src = '/stream/' + id + '?' + new Date().getTime();
        }
        setInterval(() => refreshImage(0), 100);
        setInterval(() => refreshImage(1), 100);
        document.getElementById('stream0').onerror = function() { setTimeout(() => refreshImage(0), 1000); };
        document.getElementById('stream1').onerror = function() { setTimeout(() => refreshImage(1), 1000); };
    </script>
</body>
</html>
        )";
        return crow::response(200, html);
    });
    
    server_thread = std::make_unique<std::thread>([this, port]() {
        try {
            app->port(port).multithreaded().run();
        } catch (const std::exception& e) {
            std::cerr << "Web server error: " << e.what() << std::endl;
        }
    });
    
    server_running = true;
    std::cout << "Web server started on port " << port << std::endl;
    std::cout << "Access stream at: http://localhost:" << port << std::endl;
}

void Stream::stopWebServer() {
    if (!server_running) {
        return;
    }
    
    server_running = false;
    if (app) {
        app->stop();
    }
    
    if (server_thread && server_thread->joinable()) {
        server_thread->join();
    }
    
    std::cout << "Web server stopped" << std::endl;
}

cv::Mat Stream::combineFrames(const cv::Mat& frame1_in, const cv::Mat& frame2_in) {
    
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
    
    // Thêm timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
    cv::putText(combined, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Add FPS
    std::stringstream fps_ss;
    fps_ss << "FPS: " << std::fixed << std::setprecision(2) << fps;
    cv::putText(combined, fps_ss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    return combined;
}

void Stream::updateFrames(const cv::Mat& frame1, const cv::Mat& frame2) {
    if (!server_running) {
        return;
    }

    // Calculate FPS using a simple moving average for smoother display
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_time);
    last_update_time = now;
    if (duration.count() > 0) {
        double current_fps = 1000.0 / duration.count();
        fps = (fps * 0.9) + (current_fps * 0.1); // Simple moving average
    }

    cv::Mat combined = combineFrames(frame1, frame2);

    if (!combined.empty()) {
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 85};
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (cv::imencode(".jpg", combined, jpeg_buffer, compression_params)) {
            frame_ready = true;
        } else {
            frame_ready = false;
            std::cerr << "Failed to encode combined frame" << std::endl;
        }
    }
}

void Stream::updateFrame(const cv::Mat& frame, int camera_id) {
    if (!server_running || camera_id < 0 || camera_id >= NUM_STREAMS) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - individual_last_update_times[camera_id]);
    individual_last_update_times[camera_id] = now;
    if (duration.count() > 0) {
        double current_fps = 1000.0 / duration.count();
        individual_fps[camera_id] = (individual_fps[camera_id] * 0.9) + (current_fps * 0.1);
    }

    cv::Mat processed_frame;
    if (frame.empty()) {
        processed_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(processed_frame, "Camera " + std::to_string(camera_id) + ": Offline", cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    } else {
        cv::resize(frame, processed_frame, cv::Size(640, 480));
    }

    std::stringstream fps_ss;
    fps_ss << "FPS: " << std::fixed << std::setprecision(2) << individual_fps[camera_id];
    cv::putText(processed_frame, fps_ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::cvtColor(processed_frame, processed_frame, cv::COLOR_BGR2RGB);

    if (!processed_frame.empty()) {
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 85};
        std::lock_guard<std::mutex> lock(individual_frame_mutexes[camera_id]);
        if (cv::imencode(".jpg", processed_frame, individual_jpeg_buffers[camera_id], compression_params)) {
            individual_frames_ready[camera_id] = true;
        } else {
            individual_frames_ready[camera_id] = false;
            std::cerr << "Failed to encode frame for camera " << camera_id << std::endl;
        }
    }
}

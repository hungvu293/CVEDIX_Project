#include "osd.h"
#include <iostream>

OSD::OSD() {
    app = std::make_unique<crow::SimpleApp>();
    
    // Disable Crow INFO logs - only show warnings and errors
    crow::logger::setLogLevel(crow::LogLevel::Warning);
}

void OSD::show(cv::Mat& frame) {
    cv::imshow("OSD", frame);
}

void OSD::startWebServer(int port) {
    if (server_running) {
        std::cout << "Web server already running" << std::endl;
        return;
    }
    
    // Single endpoint for combined camera view
    CROW_ROUTE((*app), "/stream")
    ([this](const crow::request& req) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (!frame_ready || combined_jpeg_buffer.empty()) {
            return crow::response(404, "No frames available");
        }
        
        crow::response res;
        res.set_header("Content-Type", "image/jpeg");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Access-Control-Allow-Origin", "*");
        res.body = std::string(combined_jpeg_buffer.begin(), combined_jpeg_buffer.end());
        return res;
    });
    
    // Status endpoint
    CROW_ROUTE((*app), "/status")
    ([this](const crow::request& req) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        crow::json::wvalue status;
        status["frame_ready"] = frame_ready;
        status["server_status"] = "running";
        return status;
    });
    
    // HTML page for viewing the stream
    CROW_ROUTE((*app), "/")
    ([](const crow::request& req) {
        std::string html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Camera Stream</title>
    <style>
        body { 
            margin: 0; 
            padding: 20px; 
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        .container { 
            text-align: center; 
            max-width: 1200px;
            margin: 0 auto;
        }
        img { 
            max-width: 100%; 
            height: auto; 
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .refresh-info {
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dual Camera Stream</h1>
        <img id="stream" src="/stream" alt="Camera Stream">
        <div class="refresh-info">
            Stream refreshes automatically every 100ms
        </div>
    </div>
    <script>
        function refreshImage() {
            const img = document.getElementById('stream');
            const timestamp = new Date().getTime();
            img.src = '/stream?' + timestamp;
        }
        
        // Refresh every 100ms for smooth streaming
        setInterval(refreshImage, 100);
        
        // Handle image load errors
        document.getElementById('stream').onerror = function() {
            setTimeout(refreshImage, 1000);
        };
    </script>
</body>
</html>
        )";
        return crow::response(200, html);
    });
    
    // Start server in separate thread
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

void OSD::stopWebServer() {
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

cv::Mat OSD::combineFrames(const std::array<cv::Mat, 2>& frames, const std::array<bool, 2>& camera_status) {
    cv::Mat combined;
    cv::Mat frame0, frame1;
    
    // Prepare frame 0
    if (camera_status[0] && !frames[0].empty()) {
        cv::resize(frames[0], frame0, cv::Size(640, 480));
    } else {
        // Create placeholder for camera 0
        frame0 = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame0, "Camera 0: Offline", 
                   cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, 
                   cv::Scalar(0, 0, 255), 2);
    }
    
    // Prepare frame 1
    if (camera_status[1] && !frames[1].empty()) {
        cv::resize(frames[1], frame1, cv::Size(640, 480));
    } else {
        // Create placeholder for camera 1
        frame1 = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame1, "Camera 1: Offline", 
                   cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, 
                   cv::Scalar(0, 0, 255), 2);
    }
    
    // Combine frames horizontally
    cv::hconcat(frame0, frame1, combined);
    
    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    cv::putText(combined, ss.str(), cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return combined;
}

void OSD::updateFrames(const std::array<cv::Mat, 2>& frames, const std::array<bool, 2>& camera_status) {
    if (!server_running) {
        return;
    }
    
    cv::Mat combined = combineFrames(frames, camera_status);
    
    if (!combined.empty()) {
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(85); // Good quality for web streaming
        
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (cv::imencode(".jpg", combined, combined_jpeg_buffer, compression_params)) {
            frame_ready = true;
        } else {
            frame_ready = false;
            std::cerr << "Failed to encode combined frame" << std::endl;
        }
    }
}

void OSD::release() {
    stopWebServer();
    cv::destroyAllWindows();
}

OSD::~OSD() {
    release();
}
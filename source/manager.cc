#include "manager.h"

#include <cassert>
#include <iostream>

#include <ctime>    // Cho std::time_t, std::localtime
#include <iomanip>  // Cho std::put_time


Pipeline::Pipeline() {}

int Pipeline::initialize(const std::vector<std::string>& rtsp_urls) {
    int ret;
    int cam_nb = rtsp_urls.size();
    if (cam_nb > 2) {
        std::cerr << "Only 2 cams max" << std::endl;
        assert(cam_nb <= 2);
    }
    for (int i = 0; i < cam_nb; i++) {
        info.rtsp_urls[i] = rtsp_urls[i];
        info.rtsp_titles[i] = "Camera " + std::to_string(i);
    }

    // double targetFps = 5.0; // Set desired frame rate (adjust as needed)
    for (int i = 0; i < cam_nb; i++) {
        readers[i] = std::make_unique<Reader>();
        // ret = readers[i]->open(rtsp_urls[i], targetFps);
        ret = readers[i]->open(rtsp_urls[i]);
        if (ret != 0) {
            std::cerr << "Init cam false: " << rtsp_urls[i] << std::endl;
            is_camera_running[i].store(false);
        }
        else {
            std::cout << "Init cam success: " << rtsp_urls[i] << std::endl;
            is_camera_running[i].store(true);
        }
    }

    // const char* model_path = "../model/yolov8.rknn";
    const char* model_path = "../model/yolo11n_plate_int8_3566_optimize.rknn";
    threadNum = 2;
    pool = std::make_unique<rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata>>(model_path, threadNum);
    if (pool->init() != 0) {
        std::cerr << "Init model pool false" << std::endl;
        return -1;
    }

    for (int i = 0; i < cam_nb; i++) {
        trackers[i] = std::make_unique<Tracking>();
    }

    std::cout << "Init success" << std::endl;
    return 0;
}

void Pipeline::start() {    
    is_system_running.store(true);
    for (int i = 0; i < is_camera_running.size(); i++) {
        decode_threads[i] = std::thread(&Pipeline::decodeLoop, this, i);
    }
    detector_thread = std::thread(&Pipeline::detectPoolLoop, this);
    for (int i = 0; i < is_camera_running.size(); i++) {
        tracking_threads[i] = std::thread(&Pipeline::trackLoop, this, i);
    }
    display_thread = std::thread(&Pipeline::displayLoop, this);
    
}

void Pipeline::stop() {
    is_system_running.store(false);
    for (int i = 0; i < is_camera_running.size(); i++) {
        is_camera_running[i].store(false);
    }    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (auto& thread : decode_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    // while (true) {
    //     std::vector<Detection> detections;
    //     if (pool->get(detections) != 0)
    //         break;
    // }
    if (detector_thread.joinable()) {
        detector_thread.join();
    }
    for (auto& thread : tracking_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (display_thread.joinable()) {
        display_thread.join();
    }
}

void Pipeline::decodeLoop(int id) {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    int ret;

    // Remove manual frame dropping since it's now handled in Reader
    // int drop_interval = 7;
    // int drop_frame_count = 0;
    if (id == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    else if (id == 1) { 
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    while (is_system_running) {
        if (!is_camera_running[id]) {
            std::cout << "Retry connect" << info.rtsp_urls[id] << std::endl;
            ret = readers[id]->open(info.rtsp_urls[id]);
            if (ret != 0) {
                if (readers[id]->isOpened) {
                    is_camera_running[id].store(true);
                }
                else {
                    std::cerr << "Init cam false: " << info.rtsp_urls[id] << std::endl;
                    readers[id]->close();
                    is_camera_running[id].store(false);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            else {
                std::cout << "Reconnect success: " << info.rtsp_urls[id] << std::endl;
                is_camera_running[id].store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        else {
            ret = readers[id]->decodeFrame(frame);
            if (ret != 0) {
                is_camera_running[id].store(false);
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            // Remove manual frame dropping logic
            // drop_frame_count++;
            // if (drop_frame_count < drop_interval) {
            //     continue;
            // }
            // drop_frame_count = 0;

            if (!frame.empty()) {
                capture_time = std::chrono::system_clock::now();
                FrameWithMetadata frame_data(frame.clone(), capture_time, id);
                ret = pool->put(frame_data);
                if (ret != 0) {
                    std::cerr << "pool put false" << std::endl;
                    break;
                }
            }
        }
        // std::cout << "done decode: " << id << std::endl;
    }
    std::cout << "Decode thread " << id << " stopped." << std::endl;
    return;

}

void Pipeline::detectPoolLoop() {
    DetectionWithMetadata result;
    int ret;
    bool any_camera_running = false;


    while (is_system_running) {
        any_camera_running = false;
        for (int i = 0; i < is_camera_running.size(); i++) {
            if (is_camera_running[i]) {
                any_camera_running = true;
                break;
            }
        }
        if (!any_camera_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ret = pool->get(result);
        if (ret != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        // for (auto det : result.detections) {
        //     cv::rectangle(result.original_frame, det.box, det.color, 2);
        //     std::string label = det.className + ": " + std::to_string(det.confidence);
        //     cv::putText(result.original_frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, det.color, 2);
        // }
        InferenceToTrack data(result.original_frame.clone(), result.capture_time, result.detections);
        inference_to_track_queues[result.camera_id].push(data);
        std::cout << "size: " << result.detections.size() << "at: " << result.camera_id << std::endl;
        for (const auto& det : result.detections) {
            std::cout << "box" << det.box << " "
                    << "confidence: " << det.confidence << " "
                    << "class_id: " << det.class_id << std::endl;
        }
        
    }    
    std::cout << "Detection pool thread stop" << std::endl;
    return;
}

void Pipeline::trackLoop(int id) {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    std::vector<Detection> detections;
    std::vector<Eigen::RowVectorXf> res;
    int ret;
    while (is_system_running) {
        if (!is_camera_running[id]) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        std::shared_ptr<InferenceToTrack> data_ptr = inference_to_track_queues[id].try_pop();
        if (!data_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame = data_ptr->origin_frame;
        capture_time = data_ptr->capture_time;
        detections = data_ptr->detections;
        if (detections.empty()) {
            std::cout << "No detections for camera " << id << std::endl;
        }
        else {
            res = trackers[id]->run(frame, detections);
            trackers[id]->draw_tracks(frame, res);
            // std::cout << "Tracking results for camera " << id << ": " << res.size() << " tracks." << std::endl;
        }
        TrackToOSD data(frame.clone(), capture_time);
        track_to_osd_queues[id].push(data);
        // std::cout << "done track: " << id << std::endl;
    }
    std::cout << "Tracking thread " << id << " stopped." << std::endl;
    return;
}

void Pipeline::displayLoop() {
    std::array<cv::Mat, 2> frames;
    std::array<std::chrono::system_clock::time_point, 2> Tcapture;
    std::chrono::system_clock::time_point Tcurrent;
    bool any_camera_running = false;
    
    // Initialize OSD with web server
    std::unique_ptr<OSD> osd = std::make_unique<OSD>();
    osd->startWebServer(8080);
    
    while (is_system_running) {
        any_camera_running = false;
        std::array<bool, 2> camera_status = {false, false};
        
        for (int i = 0; i < is_camera_running.size(); i++) {
            if (is_camera_running[i]) {
                any_camera_running = true;
                camera_status[i] = true;
            }
        }
        
        if (!any_camera_running) {
            // Update with offline status
            osd->updateFrames(frames, camera_status);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        for (int i = 0; i < is_camera_running.size(); i++) {
            if (!is_camera_running[i]) {
                continue;
            }
            
            std::shared_ptr<TrackToOSD> data_ptr = track_to_osd_queues[i].try_pop();
            if (!data_ptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            frames[i] = data_ptr->processed_frame;
            Tcapture[i] = data_ptr->capture_time;
            Tcurrent = std::chrono::system_clock::now();            
            std::chrono::duration<double, std::milli> latency = Tcurrent - Tcapture[i];
            std::cout << "Latency for camera " << i << ": " << latency.count() << " ms" << std::endl;
        }
        
        // Update frames to web server
        osd->updateFrames(frames, camera_status);
    }
    
    osd->stopWebServer();
    std::cout << "Display thread stopped." << std::endl;
    return;
}

// void Pipeline::messageLoop() {

// }
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

    for (int i = 0; i < cam_nb; i++) {
        readers[i] = std::make_unique<Reader>();
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

    const char* model_path = "../model/yolo11n_best_model-rk3566.rknn";
    threadNum = 2;
    pool = std::make_unique<rknnPool<Detector, cv::Mat, std::vector<Detection>>>(model_path, threadNum);
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
    while (true) {
        std::vector<Detection> detections;
        if (pool->get(detections) != 0)
            break;
    }
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
    
    int empty_frame_count = 0;
    
    int ret;

    int drop_interval = 5;
    int drop_frame_count = 0;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
            drop_frame_count++;
            if (drop_frame_count < drop_interval) {
                continue;
            }
            drop_frame_count = 0;
            // std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::cout << "done decode" << std::endl;

            if (!frame.empty()) {
                capture_time = std::chrono::system_clock::now();
                ret = pool->put(frame);
                if (ret != 0) {
                    std::cerr << "pool put false" << std::endl;
                    break;
                }
                pool_info_queue.push(std::make_tuple(id, capture_time, frame));
                std::cout << "pool info size" << id << ": " <<  pool_info_queue.size() << std::endl;
            }
        }
    }
    std::cout << "Decode thread " << id << " stopped." << std::endl;
    return;

}

void Pipeline::detectPoolLoop() {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    int id;
    std::vector<Detection> detections;
    int ret;
    bool any_camera_running = false;
    int current_size = 0;

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
        
        for (int i = 0; i < is_camera_running.size(); i++) {
            if (!is_camera_running[i]) {
                continue;
            } 
        }
        auto pool_info_ptr = pool_info_queue.wait_and_pop();
        if (!pool_info_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        id = std::get<0>(*pool_info_ptr);
        capture_time = std::get<1>(*pool_info_ptr);
        frame = std::get<2>(*pool_info_ptr);

        while (true) {
            ret = pool->get(detections);
            if (ret == 0) {
                break; // Got result
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        InferenceToTrack data(frame, capture_time, detections);
        inference_to_track_queues[id].push(data);
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
        std::shared_ptr<InferenceToTrack> data_ptr = inference_to_track_queues[id].wait_and_pop();
        if (!data_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame = data_ptr->origin_frame;
        capture_time = data_ptr->capture_time;
        detections = data_ptr->detections;

        res = trackers[id]->run(frame, detections);
        trackers[id]->draw_tracks(frame, res);
        
        TrackToOSD data(frame, capture_time);
        track_to_osd_queues[id].push(data);
        auto end_track_time = std::chrono::high_resolution_clock::now();
    }
    std::cout << "Tracking thread " << id << " stopped." << std::endl;
    return;
}

void Pipeline::displayLoop() {
    std::array<cv::Mat, 2> frames;
    cv::Mat concatenated_frame;
    std::array<std::chrono::system_clock::time_point, 2> Tcapture;
    std::chrono::system_clock::time_point Tcurrent;
    bool Tbefore_is_set = false;
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

        
        for (int i = 0; i < is_camera_running.size(); i++) {
            if (!is_camera_running[i]) {
                continue;
            }
            std::shared_ptr<TrackToOSD> data_ptr = track_to_osd_queues[i].wait_and_pop();
            if (!data_ptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            frames[i] = data_ptr->processed_frame;
            Tcapture[i] = data_ptr->capture_time;
            
            Tcurrent = std::chrono::system_clock::now();
            // std::chrono::duration<double, std::milli> latency = Tcurrent - Tcapture[i];
            std::chrono::duration<double, std::milli> latency = Tcurrent - data_ptr->capture_time;
            std::cout << "Latency for camera " << i << ": " << latency.count() << " ms" << std::endl;
        }

        if (!frames[0].empty() && !frames[1].empty()) {
            cv::Mat frame0_resized, frame1_resized;
            cv::resize(frames[0], frame0_resized, cv::Size(1920, 540));
            cv::resize(frames[1], frame1_resized, cv::Size(1920, 540));
            cv::Mat concatenated_frame;
            cv::vconcat(frame0_resized, frame1_resized, concatenated_frame);
            display->show(concatenated_frame);
        }
        if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
            is_system_running.store(false);
            break;
        }

    }
    std::cout << "Display thread stopped." << std::endl;
    return;
}

// void Pipeline::messageLoop() {

// }
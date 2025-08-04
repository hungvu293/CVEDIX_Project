#include "manager.h"

#include <cassert>
#include <iostream>

#include <ctime>    // Cho std::time_t, std::localtime
#include <iomanip>  // Cho std::put_time

Pipeline::Pipeline() {}

int Pipeline::initialize(const std::vector<std::string>& rtsp_urls) {
    int cam_nb = rtsp_urls.size();
    if (cam_nb > 2) {
        std::cerr << "Only 2 cams max" << std::endl;
        assert(cam_nb <= 2);
    }
    for (int i = 0; i < cam_nb; i++) {
        info.rtsp_urls[i] = rtsp_urls[i];
        info.rtsp_titles[i] = "Camera " + std::to_string(i);
    }

    int ret;
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

    const char* model_path = "../model/yolov8.rknn";
    // detector = std::make_unique<YoloV8>();
    // ret = detector->init(model_path);
    // if (ret != 0) {
    //     std::cerr << "Init model false" << std::endl;
    //     return -1;
    // }

    int threadNum = 2;
    rknnPool<YoloV8, cv::Mat, object_detect_result_list> pool(model_path, threadNum);
    

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
    detector_thread = std::thread(&Pipeline::detectLoop, this);
    for (int i = 0; i < is_camera_running.size(); i++) {
        tracking_threads[i] = std::thread(&Pipeline::trackLoop, this, i);
    }
    display_thread = std::thread(&Pipeline::displayLoop, this);
    // message_thread = std::thread(&Pipeline::messageLoop, this);
    
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

    int drop_interval = 10;
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

            if (!frame.empty()) {
                capture_time = std::chrono::system_clock::now();
                ReaderToInference data(frame, capture_time);
                reader_to_inference_queues[id].push(data);
                std::cout << "decode push: " << id << ", : " << reader_to_inference_queues[id].size() << std::endl;
            }
        }
    }
    std::cout << "Decode thread " << id << " stopped." << std::endl;
    return;

}

void Pipeline::detectLoop() {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
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

        for (int i = 0; i < is_camera_running.size(); i++) {
            if (!is_camera_running[i]) {
                continue;
            }
            std::shared_ptr<ReaderToInference> data_ptr = reader_to_inference_queues[i].wait_and_pop();
            if (!data_ptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            frame = data_ptr->origin_frame;
            capture_time = data_ptr->capture_time;        
            ret = detector->run(frame);
            if (ret != 0) {
                std::cerr << "Detection failed for camera " << i << std::endl;
                continue;   
            }
            detector->filter_class("person");

            InferenceToTrack data(frame, capture_time, *detector->od_results);
            inference_to_track_queues[i].push(data);
        }
    }
    std::cout << "Detection thread stopped." << std::endl;
    return;
}

void Pipeline::detectPoolLoop() {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
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
}

void Pipeline::trackLoop(int id) {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    object_detect_result_list od_results;
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
        // frame = data_ptr->origin_frame;
        // capture_time = data_ptr->capture_time;
        // od_results = data_ptr->od_results;
        // if (data_ptr->origin_frame.empty())
        // {
        //     std::cout << "Empty frame detected for camera " << id << std::endl;
        //     continue;
        // }

        detections = trackers[id]->convert_output(&data_ptr->od_results);
        res = trackers[id]->run(data_ptr->origin_frame, detections);
        trackers[id]->draw_tracks(data_ptr->origin_frame, res);
        
        TrackToOSD data(data_ptr->origin_frame, data_ptr->capture_time);
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
            
            // if (frames[i].empty())
            // {
            //     std::cout << "Empty frame detected for camera " << i << std::endl;
            //     continue;
            // }
            
            Tcurrent = std::chrono::system_clock::now();
            // std::chrono::duration<double, std::milli> latency = Tcurrent - Tcapture[i];
            std::chrono::duration<double, std::milli> latency = Tcurrent - data_ptr->capture_time;
            std::cout << "Latency for camera " << i << ": " << latency.count() << " ms" << std::endl;
        }

        // if (!frames[0].empty() && !frames[1].empty()) {
        //     cv::Mat frame0_resized, frame1_resized;
        //     cv::resize(frames[0], frame0_resized, cv::Size(1920, 540));
        //     cv::resize(frames[1], frame1_resized, cv::Size(1920, 540));
        //     cv::Mat concatenated_frame;
        //     cv::vconcat(frame0_resized, frame1_resized, concatenated_frame);
        //     display->show(concatenated_frame);
        // }
        // if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
        //     is_system_running.store(false);
        //     break;
        // }

    }
    std::cout << "Display thread stopped." << std::endl;
    return;
}

// void Pipeline::messageLoop() {

// }
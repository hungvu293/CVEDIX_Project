#include "manager.h"

#include <cassert>
#include <iostream>

#include <ctime>    // Cho std::time_t, std::localtime
#include <iomanip>  // Cho std::put_time


Pipeline::Pipeline() : 
    inference_to_track_queues{lock_based_queue<InferenceToTrack>(10), lock_based_queue<InferenceToTrack>(10)},
    track_to_osd_queues{lock_based_queue<TrackToOSD>(10), lock_based_queue<TrackToOSD>(10)}
{}

int Pipeline::initialize(const std::vector<std::string>& rtsp_urls) {
    int ret;
    // for (int i = 0; i < cam_nb; i++) {
    //     info.rtsp_urls[i] = rtsp_urls[i];
    //     info.rtsp_titles[i] = "Camera " + std::to_string(i);
    // }
    int cam_nb = rtsp_urls.size();

    for (int i = 0; i < cam_nb; i++) {
        readers[i] = std::make_unique<Reader>();
        ret = readers[i]->open(rtsp_urls[i], true);
        // ret = readers[i]->open(rtsp_urls[i]);

        if (ret != 0) {
            std::cerr << "Init cam false: " << rtsp_urls[i] << std::endl;
            is_camera_running[i].store(false);
        }
        else {
            std::cout << "Init cam success: " << rtsp_urls[i] << std::endl;
            is_camera_running[i].store(true);
        }
    }

    std::string model_path = "../model/yolo11n_quantization_no_postprocessing.rknn";
    threadNum = 4;
    pool = std::make_unique<rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata>>(model_path, threadNum);
    if (pool->init() != 0) {
        std::cerr << "Init model pool false" << std::endl;
        return -1;
    }

    for (int i = 0; i < cam_nb; i++) {
        trackers[i] = std::make_unique<Tracking>();
    }

    stream = std::make_unique<Stream>();
    stream->startWebServer(8080);

    message = std::make_unique<Message>("tcp://127.0.0.1:1883", "rockchip_3566");
    message->connect();

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
    int ret;
    while (is_system_running) {
        if (!is_camera_running[id]) {
            std::cout << "Retry connect" << info.rtsp_urls[id] << std::endl;
            ret = readers[id]->open(info.rtsp_urls[id], true);
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
            cv::Mat frame;
            std::chrono::system_clock::time_point capture_time = std::chrono::system_clock::now();
            ret = readers[id]->decodeFrame(frame);
            if (ret != 0) {
                is_camera_running[id].store(false);
                continue;
            }
            
            if (!frame.empty()) {
                std::time_t t_current = std::chrono::system_clock::to_time_t(capture_time);
                FrameWithMetadata frame_data(std::move(frame), capture_time, id);
                ret = pool->put(std::move(frame_data));
                if (ret != 0) {
                    std::cout << "Pool full, skipping frame from camera " << id << std::endl;
                    continue; 
                }
            }
        }
    }
    std::cout << "Decode thread " << id << " stopped." << std::endl;
    return;

}

void Pipeline::detectPoolLoop() {
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

        DetectionWithMetadata result;
        ret = pool->get(result);
        if (ret != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        InferenceToTrack data(std::move(result.original_frame), result.capture_time, std::move(result.detections));
        inference_to_track_queues[result.camera_id].push(std::move(data));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
    }    
    std::cout << "Detection pool thread stop" << std::endl;
    return;
}

void Pipeline::trackLoop(int id) {
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
        cv::Mat frame = data_ptr->origin_frame;
        std::chrono::system_clock::time_point capture_time = data_ptr->capture_time;
        std::vector<Detection> detections = data_ptr->detections;
        
        trackers[id]->run(frame, detections);
        trackers[id]->draw_tracks(frame);

        auto plates = trackers[id]->getPlates();
        for (const auto& plate : plates) {
            cv::Rect img_rect(0, 0, frame.cols, frame.rows);
            cv::Rect valid_plate = plate & img_rect;
            if (valid_plate.width > 0 && valid_plate.height > 0) {
                cv::Mat plate_img = frame(valid_plate);
                // message->sendMessage(capture_time, id, plate_img);
            }
        }

        TrackToOSD data(std::move(frame), capture_time);
        track_to_osd_queues[id].push(std::move(data));
    }
    std::cout << "Tracking thread " << id << " stopped." << std::endl;
    return;
}

void Pipeline::displayLoop() {
    std::array<cv::Mat, 2> frames;
    std::array<std::chrono::system_clock::time_point, 2> Tcapture;
    std::chrono::system_clock::time_point Tcurrent;
    bool any_camera_running = false;
    
    while (is_system_running) {
        any_camera_running = false;
        
        for (int i = 0; i < is_camera_running.size(); i++) {
            if (is_camera_running[i]) {
                any_camera_running = true;
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
            
            stream->updateFrame(frames[i], i);

        }
    }
    std::cout << "Display thread stopped." << std::endl;
    return;
}

// void Pipeline::messageLoop() {

// }
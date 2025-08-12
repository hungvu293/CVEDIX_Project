#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

// #include "manager.h"
#include "detector.h"
#include "rknnPool.hpp"
#include "reader.h"
#include "track.h"

void producer(Reader& reader, rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata>& pool, std::atomic<bool>& done) {
    while (reader.isOpened || !done) {
        cv::Mat frame;
        if (reader.decodeFrame(frame) != 0) {
            done.store(true);
            break; // Stream kết thúc hoặc lỗi
        }
        if (frame.empty()) {
            continue;
        }
        auto capture_time = std::chrono::system_clock::now();
        int id = 0;
        FrameWithMetadata frame_data(std::move(frame), capture_time, id);
        pool.put(std::move(frame_data));
    }
    reader.close();
    std::cout << "Producer finished." << std::endl;
}

void consumer(rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata>& pool, Tracking& tracker, std::atomic<bool>& done) {
    while (!done || pool.get_queue_size() > 0) {
        DetectionWithMetadata result;
        if (pool.get(result) == 0) {
            // Lọc để chỉ giữ lại các phát hiện có class_id là 0
            std::vector<Detection> filtered_detections;
            for (const auto& det : result.detections) {
                if (det.class_id == 0) {
                    filtered_detections.push_back(det);
                }
            }

            tracker.run(result.original_frame, filtered_detections);
            tracker.draw_tracks(result.original_frame);
            cv::imshow("OSD", result.original_frame);

            if (cv::waitKey(1) >= 0) {
                done.store(true);
                break;
            }
        } else {
            // Hàng đợi rỗng, chờ một chút trước khi thử lại
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (done) break;
        }
    }
    std::cout << "Consumer finished." << std::endl;
}

int main() {
    // std::string model_path = "../model/yolo11n_plate_int8_3566_optimize.rknn";
    std::string model_path = "../model/yolov8.rknn";
    int threadNum = 4;
    rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata> pool(model_path, threadNum);
    if (pool.init() != 0) {
        std::cerr << "Failed to initialize rknnPool." << std::endl;
        return -1;
    }
    std::cout << "rknnPool initialized successfully with " << threadNum << " threads." << std::endl;

    Reader reader;
    std::string input1 = "rtsp://user03:abcd1234@113.177.126.32:8153";
    if (reader.open(input1, true) != 0) return -1;
    // if (reader.open(input1, false) != 0) return -1;


    Tracking tracker;
    
    std::atomic<bool> done(false);

    // Khởi chạy luồng producer và consumer
    std::thread producer_thread(producer, std::ref(reader), std::ref(pool), std::ref(done));
    std::thread consumer_thread(consumer, std::ref(pool), std::ref(tracker), std::ref(done));

    // Chờ các luồng hoàn thành
    producer_thread.join();
    consumer_thread.join();

    reader.close();
    cv::destroyAllWindows();
    return 0;
}

// int main() {
//     /////////////
//     std::string input0 =  "rtsp://103.147.186.175:8554/9L02DA3PAJ39B2F";
//     std::string input1 = "rtsp://user03:abcd1234@113.177.126.32:8153";
//     // std::string input2 = "rtsp://admin:Admin123456@192.168.1.200:8554/cam/realmonitor?channel=1&subtype=0";
//     std::vector<std::string> input = {input0, input1};
//     Pipeline pipeline;
//     int ret = pipeline.initialize(input);
//     if (ret != 0) {
//         std::cerr << "Pipeline initialization failed." << std::endl;
//         return -1;
//     }
//     pipeline.start();
    
//     std::string line;
//     std::cout << "Press Enter to stop the pipeline..." << std::endl;
//     std::getline(std::cin, line);
//     pipeline.stop();
// }
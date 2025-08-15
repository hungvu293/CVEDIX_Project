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
#include "stream.h"
#include "osd.h" 
#include "server.h"
#include "message.h"

void producer(Reader& reader, rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata>& pool, std::atomic<bool>& done, int reader_id) {
    while (!done) {
        cv::Mat frame;
        if (reader.decodeFrame(frame) != 0) {
            std::cout << "Producer " << reader_id << " stream ended or error." << std::endl;
            break; // Stream kết thúc hoặc lỗi
        }
        if (frame.empty()) {
            continue;
        }
        auto capture_time = std::chrono::system_clock::now();
        FrameWithMetadata frame_data(std::move(frame), capture_time, reader_id);
        pool.put(std::move(frame_data));
    }
    std::cout << "Producer " << reader_id << " finished." << std::endl;
}

void consumer(rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata>& pool, Tracking& tracker0, Tracking& tracker1, Stream& stream, OSD& osd, Server& server, Message& message_client, std::atomic<bool>& done) {
    cv::Mat frame0, frame1;
    std::mutex frame_mutex;
    while (!done) {
        DetectionWithMetadata result;
        if (pool.get(result) == 0) {
            // Lọc để chỉ giữ lại các phát hiện có class_id là 0
            std::vector<Detection> filtered_detections;
            for (auto det : result.detections) { // Use a copy to modify it
                if (det.class_id == 0) {
                    // Increase width and height by 30%
                    float scale = 1.7f;
                    int new_width = static_cast<int>(det.box.width * scale);
                    int new_height = static_cast<int>(det.box.height * scale);

                    // Adjust x and y to keep the box centered
                    det.box.x -= (new_width - det.box.width) / 2;
                    det.box.y -= (new_height - det.box.height) / 2;
                    det.box.width = new_width;
                    det.box.height = new_height;
                    
                    filtered_detections.push_back(det);
                }
            }

            for (const auto& det : filtered_detections) {
                cv::rectangle(result.original_frame, det.box, det.color, 2);
                std::string label = det.className + ": " + std::to_string(det.confidence);
                cv::putText(result.original_frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, det.color, 2);
            }

            if (result.camera_id == 0) {
                tracker0.run(result.original_frame, filtered_detections);
                // std::vector<cv::Rect> plates0 = tracker0.getPlates();
                // for (const auto& plate : plates0) {
                //     // Ensure the plate rectangle is within the frame boundaries before cropping
                //     cv::Rect img_rect(0, 0, result.original_frame.cols, result.original_frame.rows);
                //     cv::Rect valid_plate = plate & img_rect;
                //     if (valid_plate.width > 0 && valid_plate.height > 0) {
                //         cv::Mat plate_img = result.original_frame(valid_plate);
                //         message_client.sendMessage(result.capture_time, result.camera_id, plate_img);
                //     }
                // }
                tracker0.draw_tracks(result.original_frame);
                stream.updateFrame(result.original_frame, 0);
                // server.encodeFrame(result.original_frame);
            }
            else if (result.camera_id == 1) {
                tracker1.run(result.original_frame, filtered_detections);
                // std::vector<cv::Rect> plates1 = tracker1.getPlates();
                // for (const auto& plate : plates1) {
                //     // Ensure the plate rectangle is within the frame boundaries before cropping
                //     cv::Rect img_rect(0, 0, result.original_frame.cols, result.original_frame.rows);
                //     cv::Rect valid_plate = plate & img_rect;
                //     if (valid_plate.width > 0 && valid_plate.height > 0) {
                //         cv::Mat plate_img = result.original_frame(valid_plate);
                //         message_client.sendMessage(result.capture_time, result.camera_id, plate_img);
                //     }
                // }
                tracker1.draw_tracks(result.original_frame);
                stream.updateFrame(result.original_frame, 1);
            }

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

void handle_input(std::atomic<bool>& done) {
    std::cout << "Press Enter to stop..." << std::endl;
    std::string line;
    std::getline(std::cin, line);
    done.store(true);
}

int main() {
    // Tạo 2 Reader objects
    Reader reader0, reader1;
    
    std::string input0 = "rtsp://user03:abcd1234@113.177.128.13:8159";
    std::string input1 = "rtsp://user03:abcd1234@113.177.126.32:8153";
    
    // Mở cả 2 streams
    if (reader0.open(input0) != 0) {
        std::cerr << "Failed to open input0: " << input0 << std::endl;
        return -1;
    }
    
    // if (reader1.open(input1, true) != 0) {
    //     std::cerr << "Failed to open input1: " << input1 << std::endl;
    //     reader0.close();
    //     return -1;
    // }

    // std::string model_path = "../model/yolo11n_plate_int8_3566_optimize.rknn";
    std::string model_path = "../model/yolo11n_quantization_no_postprocessing.rknn";
    // std::string model_path = "../model/yolov8.rknn";
    int threadNum = 6;
    rknnPool<Detector, FrameWithMetadata, DetectionWithMetadata> pool(model_path, threadNum);
    if (pool.init() != 0) {
        std::cerr << "Failed to initialize rknnPool." << std::endl;
        return -1;
    }
    std::cout << "rknnPool initialized successfully with " << threadNum << " threads." << std::endl;

    Tracking tracker0, tracker1;
    Stream stream; // MJPEG Stream
    stream.startWebServer(8080); 

    OSD osd;

    Message message_client("tcp://127.0.0.1:1883", "rockchip_3566");
    message_client.connect();

    Server server; // RTSP Server
    // std::string rtsp_url = "rtsp://103.147.186.216:20990/livestream/00_11_22_33_44/index.rtsp?username=1124a9a7183d0257842986fe3e83fc21&time=UTC&key=&token=";
    // if (server.open(rtsp_url, 1600, 1200, 6, false) != 0) {
    //     std::cerr << "Failed to open RTSP server" << std::endl;
    // }
    
    std::atomic<bool> done(false);

    // Luồng riêng để xử lý input
    std::thread input_thread(handle_input, std::ref(done));

    // Khởi chạy 2 luồng producer và 1 luồng consumer
    std::thread producer_thread0(producer, std::ref(reader0), std::ref(pool), std::ref(done), 0);
    // std::thread producer_thread1(producer, std::ref(reader1), std::ref(pool), std::ref(done), 1);
    std::thread consumer_thread(consumer, std::ref(pool), std::ref(tracker0), std::ref(tracker1), std::ref(stream), std::ref(osd), std::ref(server), std::ref(message_client), std::ref(done));

    // Chờ các luồng hoàn thành
    // producer_thread0.join();
    // producer_thread1.join();
    consumer_thread.join();

    // Đảm bảo luồng input cũng kết thúc
    input_thread.join();

    // Đảm bảo đóng các readers
    reader0.close();
    reader1.close();

    stream.stopWebServer(); // Dừng server
    message_client.disconnect();
    server.close();
    osd.release();
    // cv::destroyAllWindows();
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
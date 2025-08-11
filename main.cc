#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include "manager.h"
// #include "detector.h"
// #include <reader.h>

int main() {
    /////////////
    std::string input0 =  "rtsp://103.147.186.175:8554/9L02DA3PAJ39B2F";
    std::string input1 = "rtsp://user03:abcd1234@113.177.126.32:8153";
    // std::string input2 = "rtsp://admin:Admin123456@192.168.1.200:8554/cam/realmonitor?channel=1&subtype=0";
    std::vector<std::string> input = {input0, input1};
    Pipeline pipeline;
    int ret = pipeline.initialize(input);
    if (ret != 0) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return -1;
    }
    pipeline.start();
    
    std::string line;
    std::cout << "Press Enter to stop the pipeline..." << std::endl;
    std::getline(std::cin, line);
    pipeline.stop();
}

// int main() {
//     std::string model_path = "../model/yolo11n_plate_int8_3566_optimize.rknn";
//     Detector detector(model_path);
//     if (detector.init(nullptr, false) != 0) {
//         std::cerr << "Failed to initialize detector." << std::endl;
//         return -1;
//     }
//     cv::Mat img = cv::imread("../plate.png");
//     if (img.empty()) {
//         std::cerr << "Failed to load plate.png" << std::endl;
//         return -1;
//     }
//     auto start = std::chrono::steady_clock::now();
//     std::vector<Detection> detections = detector.infer(img);
//     detector.draw(img, detections);

//     auto end = std::chrono::steady_clock::now();
//     std::chrono::duration<double, std::milli> duration = end - start;
//     std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
//     if (!cv::imwrite("../out_plate.jpg", img)) {
//         std::cerr << "Failed to save out_plate.jpg" << std::endl;
//         return -1;
//     }
//     std::cout << "Detection and drawing done. Saved to out_plate.jpg" << std::endl;
//     return 0;
// }


// void run_reader(const std::string url) {
//     Reader r;
//     if (r.open(url) < 0) return; // Default to software decoder
//     cv::Mat frame;
//     while (true) {
//         int ret = r.decodeFrame(frame);
//         if (ret < 0) break;
//     }
//     r.close();
// }

// int main() {
//     const std::string url1 = "rtsp://user03:abcd1234@113.177.126.84:8202";
//     const std::string url2 = "rtsp://103.147.186.175:8554/9L02DA3PAJ39B2F";

//     std::thread t1(run_reader, url2);
//     std::thread t2(run_reader, url2);

//     t1.join();
//     t2.join();
//     return 0;
// }
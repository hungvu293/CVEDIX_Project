#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>

std::queue<cv::Mat> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCond;
std::atomic<bool> isRunning = true;

cv::VideoCapture init(const std::string& source) {
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr <<"VideoCapture not opened"<<std::endl;
        isRunning = false;
        queueCond.notify_all();
        exit(-1);
    }
    return cap;
}

void source(cv::VideoCapture& cap){
    cv::Mat frame;
    while (isRunning) {
        cap.read(frame);
        if (frame.empty()) {
            if (!cap.isOpened() || (cap.get(cv::CAP_PROP_POS_FRAMES) > 0 && cap.get(cv::CAP_PROP_POS_FRAMES) == cap.get(cv::CAP_PROP_FRAME_COUNT))) {
                std::cout << "End of video stream or consistent empty frame. Stopping source thread." << std::endl;
                isRunning = false; 
                queueCond.notify_all();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
            continue;
        }   
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(frame.clone());
            std::cout << "Queue size after push " << frameQueue.size() << std::endl;
        }
        queueCond.notify_all();
        
        if (!isRunning){
            break;
        }
    }
    cap.release();
}

void osd(){
    cv::Mat frame;
    while (isRunning) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCond.wait(lock, [] {
            return !frameQueue.empty() || !isRunning;
        });
        if (frameQueue.empty() && !isRunning) {
            break;
        }
        frame = frameQueue.front();
        frameQueue.pop();
        std::cout << "Queue size after pop " << frameQueue.size() << std::endl;
        if (!frame.empty()) {
            cv::imshow("receiver", frame);
        }
        else {
            continue;
        }
        if (cv::waitKey(1) == 27) {
            isRunning = false;
            queueCond.notify_all();
        }
    }
    cv::destroyAllWindows();
}

void inference(cv::Mat& raw_frame){}

int main() {
    // std::string input = "/dev/video0";
    std::string input = "rtsp://103.147.186.175:8554/9L02DA3PAJ39B2F";
    cv::VideoCapture cap = init(input);

    std::thread source_t(source, std::ref(cap));
    std::thread osd_t(osd);
    source_t.join();
    osd_t.join();

    // cv::Mat frame;
    // while (true) {
    //     cap.read(frame);
    //     if (frame.empty()) {
    //         break;
    //     }
    //     else {
    //         cv::imshow("sync",frame);
    //     }
    //     if (cv::waitKey(1) == 27) {
    //         break;
    //     }
    // }
    // cap.release();
    // cv::destroyAllWindows();

    return 0;
}

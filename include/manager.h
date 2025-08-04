#ifndef MANAGER_H
#define MANAGER_H

#include <array>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>

#include "rknnPool.hpp"

#include "manager.h"
#include "data.h"
#include "yolov8.h"
#include "track.h"
#include "reader.h"
#include "osd.h"
// #include "mqtt"

struct SystemStat {
    std::array<std::atomic<double>, 2> fps;
    std::array<std::atomic<double>, 2> latency;
};

struct ReaderInfo {
    std::array<std::string, 2> rtsp_urls;
    std::array<std::string, 2> rtsp_titles;
};

class Pipeline {
private:
    // Object holders
    std::array<std::unique_ptr<Reader>, 2> readers;
    std::unique_ptr<YoloV8> detector;
    std::array<std::unique_ptr<Tracking>, 2> trackers;
    std::unique_ptr<OSD> display;
    // std::unique_ptr<MQTT> message;

    //Data holders
    std::array<lock_based_queue<ReaderToInference>, 2> reader_to_inference_queues;
    std::array<lock_based_queue<InferenceToTrack>, 2> inference_to_track_queues;
    std::array<lock_based_queue<TrackToOSD>, 2> track_to_osd_queues;
    //MQTT message

    //Thread holders
    std::array<std::thread, 2> decode_threads;
    std::thread detector_thread;
    std::array<std::thread, 2> tracking_threads;
    std::thread display_thread;
    // std::thread message_thread

    //Flag
    std::atomic<bool> is_system_running;
    std::array<std::atomic<bool>, 2> is_camera_running;

    SystemStat stat;
    ReaderInfo info;


public:
    Pipeline();
    int initialize(const std::vector<std::string>& rtsp_urls);
    void start();
    void stop();
    void printStats();

private:
    void decodeLoop(int id);
    void detectLoop();
    void detectPoolLop();
    void trackLoop(int id);
    void displayLoop();
    void messageLoop();
};

#endif
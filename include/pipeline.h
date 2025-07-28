#ifndef PIPELINE_H
#define PIPELINE_H

#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>
#include <string>

#include <pipeline.h>
#include <data.h>
#include <yolov8.h>
#include <reader.h>
#include <osd.h>


void img_inference(const char* model_path, const char* input);
void sync(const char* model_path, std::string& input);
void async(const char* model_path, std::string& input);

void read_thread_func(Reader& reader, 
    lock_free_queue<ReaderToInference>& output_queue, 
    std::atomic<bool>& running);
void inference_thread_func(YoloV8& inference, 
    lock_free_queue<ReaderToInference>& input_queue, lock_free_queue<InferenceToTrack>& output_queue, 
    std::atomic<bool>& running);
void track_thread_func(Tracking& track, 
    lock_free_queue<InferenceToTrack>& input_queue, lock_free_queue<TrackToOSD>& output_queue, 
    std::atomic<bool>& running);
void display_thread_func(OSD& osd, 
    lock_free_queue<TrackToOSD>& input_queue, 
    std::atomic<bool>& running);
#endif 


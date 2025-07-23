#include <thread>
#include <atomic>
#include <iostream>
#include <string>

#include <chrono>   // Để dùng std::chrono::system_clock::time_point
#include <iomanip>  // Để dùng std::put_time
#include <ctime>    // Để dùng std::time_t, std::localtime, std::gmtime


#include "pipeline.h"
#include "data.h"
#include "yolov8.h"
#include "reader.h"
#include "osd.h"

void img_inference(const char* model_path, const char* img_path) {
    YoloV8 inference;
    int ret;
    std::chrono::steady_clock::time_point Tbegin, Tend;

    // Init
    ret = inference.init(model_path);
    if (ret != 0) {
    }
    cv::Mat orig_img;

    // Input
    orig_img=cv::imread(img_path, 1);
    if (orig_img.empty())
    {
        printf("Error grabbing img\n");
    }
    
    //Inference
    ret = inference.run(orig_img);
    if (ret != 0)
    {
        printf("inference fail\n");
    }

    //Output
    ret = inference.draw(orig_img);
    if (ret != 0)
    {
        printf("draw fail\n");
    }
    imwrite("../out.jpg", orig_img);
}

void sync(const char* model_path, std::string& input) {
    Reader reader;
    YoloV8 inference;
    OSD osd;
    int ret;
    cv::Mat orig_img;
    std::chrono::steady_clock::time_point Tbegin, Tend;
    
    // Init
    ret = reader.init(input);
    if (ret != 0) {
        goto out;
    }

    ret = inference.init(model_path);
    if (ret != 0) {
        goto out;
    }
    // Inference
    while (true) {
        Tbegin = std::chrono::steady_clock::now();
        reader.read(orig_img);
        if (orig_img.empty()) {
            break;
        }
        else {
            auto start_inference_time = std::chrono::high_resolution_clock::now();
            ret = inference.run(orig_img);
            if (ret != 0) {
                break;
            }
            ret = inference.draw(orig_img);
            if (ret != 0) {
                break;
            }
            auto end_inference_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inference_duration = end_inference_time - start_inference_time;
            std::cout << "Model inference time: " << inference_duration.count() << " ms" << std::endl;
            // Draw FPS
            Tend = std::chrono::steady_clock::now();
            double duration_s = std::chrono::duration_cast<std::chrono::microseconds>(Tend - Tbegin).count() / 1000000.0;
            double fps = 1.0 / duration_s;
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << fps;
            std::string fps_text = "FPS: " + ss.str(); 
            cv::putText(orig_img, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // Display
            osd.show(orig_img);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    }
    goto out;
out:
    reader.release();
    inference.release();
    osd.release();
    return;
}

void async(const char* model_path, std::string& input) {
    Reader reader;
    YoloV8 inference;
    OSD osd;
    // threadsafe_queue<cv::Mat> reader_to_inference_queue;
    // threadsafe_queue<cv::Mat> inference_to_osd_queue;
    threadsafe_queue<ReaderToInference> reader_to_inference_queue;
    threadsafe_queue<InferenceToOSD> inference_to_osd_queue;
    std::atomic<bool> running = true;

    int ret;
    // Init
    ret = reader.init(input);
    if (ret != 0) {
    }

    ret = inference.init(model_path);
    if (ret != 0) {
    }
    // osd.init_display();

    std::thread read_t(read_thread_func, std::ref(reader), std::ref(reader_to_inference_queue), std::ref(running));
    std::thread run_t(run_thread_func, std::ref(inference), std::ref(reader_to_inference_queue), std::ref(inference_to_osd_queue), std::ref(running));
    std::thread display_t(display_thread_func, std::ref(osd), std::ref(inference_to_osd_queue), std::ref(running));

    read_t.join();
    run_t.join();
    display_t.join();
    return;
}

void read_thread_func(Reader& reader, threadsafe_queue<ReaderToInference>& output_queue, std::atomic<bool>& running) {
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    int drop_count = 4;
    int i = 0;
    while (running) {
        i++;
        reader.read(frame);
        if (i % drop_count == 0) {
            capture_time = std::chrono::system_clock::now();
            // std::cout << "Reader: " << format_system_time_point(capture_time) << std::endl;
            if (frame.empty()) {
                if (!reader.isInitialized) {
                    running = false;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
                continue;
            }
            output_queue.push(ReaderToInference(frame, capture_time));
            std::cout << "first: " << output_queue.size() << std::endl;

            i = 0;
        }

        if (!running) 
            break;
    }
        reader.release();
        return;
}


void run_thread_func(YoloV8& inference, threadsafe_queue<ReaderToInference>& input_queue, threadsafe_queue<InferenceToOSD>& output_queue, 
std::atomic<bool>& running) {
    ReaderToInference data;
    cv::Mat frame;
    std::chrono::system_clock::time_point capture_time;
    int ret;
    while (running) {
        std::shared_ptr<ReaderToInference> data_ptr = input_queue.try_pop();
        if (!data_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        data = *data_ptr;
        frame = data.origin_frame;
        capture_time = data.capture_time;
        auto start_inference_time = std::chrono::high_resolution_clock::now();
        ret = inference.run(frame);
        if (ret != 0) {
            break;
        }
        auto start_draw = std::chrono::high_resolution_clock::now();
        ret = inference.draw(frame);
        if (ret != 0) {
            break;
        }
        auto end_inference_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_duration = end_inference_time - start_inference_time;
        std::chrono::duration<double, std::milli> draw_duration = end_inference_time - start_draw;
        std::cout << "Model inference time: " << inference_duration.count() << " ms" << std::endl;
        std::cout << "Model draw time: " << draw_duration.count() << " ms" << std::endl;
        output_queue.push(
            InferenceToOSD(frame, capture_time)
        );
        if (!running)
            break;
    }
    inference.release();
    return;
}

void display_thread_func(OSD& osd, threadsafe_queue<InferenceToOSD>& input_queue, std::atomic<bool>& running) {
    InferenceToOSD data;
    cv::Mat frame;
    std::chrono::system_clock::time_point Tcapture, Tcurrent, Tbefore;
    bool Tbefore_is_set = false;
    while (running) {
        std::shared_ptr<InferenceToOSD> data_ptr = input_queue.try_pop();
        if (!data_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        data = *data_ptr;
        frame = data.processed_frame;
        Tcapture = data.capture_time;

        std::cout << "second: " << input_queue.size() << std::endl;

        // std::cout << "Reader (from display): " << format_system_time_point(Tcapture) << std::endl;
        Tcurrent = std::chrono::system_clock::now();
        // std::cout << "Current display: " << format_system_time_point(Tcurrent) << std::endl;
        double delay = std::chrono::duration_cast<std::chrono::microseconds>(Tcurrent - Tcapture).count() / 1000000.0;
        std::stringstream delay_stream;
        delay_stream << std::fixed << std::setprecision(2) << delay;
        std::string delay_text = "Delay: " + delay_stream.str() + "s";
        cv::putText(frame, delay_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        if (Tbefore_is_set) {
            double duration_s = std::chrono::duration_cast<std::chrono::microseconds>(Tcurrent - Tbefore).count() / 1000000.0;
            double fps = 1.0 / duration_s;
            std::stringstream fps_stream;
            fps_stream << std::fixed << std::setprecision(2) << fps;
            std::string fps_text = "FPS: " + fps_stream.str(); 
            cv::putText(frame, fps_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        Tbefore = std::chrono::system_clock::now();
        if (!Tbefore_is_set)
            Tbefore_is_set = true;

            
        osd.show(frame);
        if (cv::waitKey(1) == 27) {
            running = false;
            break;
        }

        if (!running)
            break;
    }
    osd.release();
    return;

}
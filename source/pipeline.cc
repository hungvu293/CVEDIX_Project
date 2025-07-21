#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>

#include <pipeline.h>
#include <yolov8.h>
#include <reader.h>
#include <osd.h>


template<typename T>
threadsafe_queue<T>::threadsafe_queue() {}

template<typename T>
void threadsafe_queue<T>::wait_and_pop(T& value) {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this] {return !data_queue.empty();});
    value = std::move(*data_queue.front()); 
    data_queue.pop();
}

template<typename T>
bool threadsafe_queue<T>::try_pop(T& value) {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty())
        return false;
    value = std::move(*data_queue.front());
    data_queue.pop();
    return true;
}

template<typename T>
std::shared_ptr<T> threadsafe_queue<T>::wait_and_pop() {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this] {return !data_queue.empty();});
    std::shared_ptr<T> res = data_queue.front(); 
    data_queue.pop();
    return res; 
}

template<typename T>
std::shared_ptr<T> threadsafe_queue<T>::try_pop() {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty())
        return std::shared_ptr<T>(); 
    std::shared_ptr<T> res = data_queue.front();
    data_queue.pop();
    return res;
}

template<typename T>
void threadsafe_queue<T>::push(T new_value) {
    std::shared_ptr<T> data = std::make_shared<T>(std::move(new_value));

    std::lock_guard<std::mutex> lk(mut); 
    data_queue.push(data);               
    data_cond.notify_one();              
}

template<typename T>
bool threadsafe_queue<T>::empty() const {
    std::lock_guard<std::mutex> lk(mut); 
    return data_queue.empty();
}

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

void sync(const char* model_path, const char* input) {
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
            ret = inference.run(orig_img);
            if (ret != 0) {
                break;
            }
            ret = inference.draw(orig_img);
            if (ret != 0) {
                break;
            }
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

void read_thread_func(Reader& reader, threadsafe_queue<cv::Mat>& output_queue, std::atomic<bool>& running);
void run_thread_func(YoloV8& inference, threadsafe_queue<cv::Mat>& input_queue, threadsafe_queue<cv::Mat>& output_queue, 
std::atomic<bool>& running);
void display_thread_func(OSD& osd, threadsafe_queue<cv::Mat>& input_queue, std::atomic<bool>& running);

void async(const char* model_path, const char* input) {
    Reader reader;
    YoloV8 inference;
    OSD osd;
    threadsafe_queue<cv::Mat> reader_to_inference_queue;
    threadsafe_queue<cv::Mat> inference_to_osd_queue;
    std::atomic<bool> running = true;

    int ret;
    // Init
    ret = reader.init(input);
    if (ret != 0) {
    }

    ret = inference.init(model_path);
    if (ret != 0) {
    }
    std::thread read_t(read_thread_func, std::ref(reader), std::ref(reader_to_inference_queue), std::ref(running));
    std::thread run_t(run_thread_func, std::ref(inference), std::ref(reader_to_inference_queue), std::ref(inference_to_osd_queue), std::ref(running));
    std::thread display_t(display_thread_func, std::ref(osd), std::ref(inference_to_osd_queue), std::ref(running));

    read_t.join();
    run_t.join();
    display_t.join();
    return;
}

void read_thread_func(Reader& reader, threadsafe_queue<cv::Mat>& output_queue, std::atomic<bool>& running) {
    cv::Mat frame;
    while (running) {
        reader.read(frame);
        if (frame.empty()) {
            if (!reader.isInitialized) {
                running = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
            continue;
        }
        output_queue.push(frame);
        if (!running) 
            break;
    }
        reader.release();
        return;
}


void run_thread_func(YoloV8& inference, threadsafe_queue<cv::Mat>& input_queue, threadsafe_queue<cv::Mat>& output_queue, 
std::atomic<bool>& running) {
    cv::Mat frame;
    int ret;
    while (running) {
        std::shared_ptr<cv::Mat> frame_ptr = input_queue.try_pop();
        if (!frame_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame = *frame_ptr;
        ret = inference.run(frame);
        if (ret != 0) {
            break;
        }
        ret = inference.draw(frame);
        if (ret != 0) {
            break;
        }
        output_queue.push(frame);
        if (!running)
            break;
    }
    inference.release();
    return;
}

void display_thread_func(OSD& osd, threadsafe_queue<cv::Mat>& input_queue, std::atomic<bool>& running) {
    cv::Mat frame;
    while (running) {
        std::shared_ptr<cv::Mat> frame_ptr = input_queue.try_pop();
        if (!frame_ptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame = *frame_ptr;
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
#ifndef DATA_H
#define DATA_H

#include <queue>              
#include <mutex>              
#include <condition_variable> 
#include <memory>             
#include <utility>   
#include <chrono>

#include "opencv2/opencv.hpp"

template<typename T>
class threadsafe_queue {
private:
    mutable std::mutex mut;
    std::queue<std::shared_ptr<T>> data_queue;
    std::condition_variable data_cond;
public:
    threadsafe_queue();
    void wait_and_pop(T& value);
    bool try_pop(T& value);
    std::shared_ptr<T> wait_and_pop();
    std::shared_ptr<T> try_pop();
    void push(T new_value);
    bool empty() const;
    int size();

};

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

template<typename T>
int threadsafe_queue<T>::size() {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.size();
}


struct ReaderToInference {
    cv::Mat origin_frame;
    std::chrono::system_clock::time_point capture_time;
    ReaderToInference(cv::Mat& frame, std::chrono::system_clock::time_point time) : origin_frame(frame.clone()), capture_time(time) {};
    ReaderToInference(): origin_frame(), capture_time() {};
};

struct InferenceToOSD {
    cv::Mat processed_frame;
    std::chrono::system_clock::time_point capture_time;
    InferenceToOSD(cv::Mat& frame, std::chrono::system_clock::time_point time) : processed_frame(frame.clone()), capture_time(time) {};
    InferenceToOSD(): processed_frame(), capture_time() {};
};

#endif 
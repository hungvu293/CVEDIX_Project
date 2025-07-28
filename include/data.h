#ifndef DATA_H
#define DATA_H

#include <queue>              
#include <mutex>              
#include <condition_variable> 
#include <memory>             
#include <utility>   

#include <chrono>
#include "opencv2/opencv.hpp"

#include "yolov8.h"
#include "track.h"

template<typename T>
class lock_based_queue {
private:
    mutable std::mutex mut;
    std::queue<std::shared_ptr<T>> data_queue;
    std::condition_variable data_cond;

public:
    lock_based_queue() = default;

    lock_based_queue(const lock_based_queue& other) = delete;
    lock_based_queue& operator=(const lock_based_queue& other) = delete;

    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        value = std::move(*data_queue.front());
        data_queue.pop();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) {
            return false;
        }
        value = std::move(*data_queue.front());
        data_queue.pop();
        return true;
    }

    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        std::shared_ptr<T> res = data_queue.front();
        data_queue.pop();
        return res;
    }

    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> res = data_queue.front();
        data_queue.pop();
        return res;
    }

    void push(T new_value) {
        std::shared_ptr<T> data = std::make_shared<T>(std::move(new_value));
        std::lock_guard<std::mutex> lk(mut);
        data_queue.push(data);
        data_cond.notify_one();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }

    int size() {
        std::lock_guard<std::mutex> lk(mut);
        return static_cast<int>(data_queue.size());
    }
};

template<typename T>
class lock_free_queue {
private:
    struct node {
        std::shared_ptr<T> data;
        node* next;
        node():
            next(nullptr)
        {}
    };
    std::atomic<node*> head;
    std::atomic<node*> tail;
    node* pop_head() {
        node* const old_head=head.load();
        if (old_head == tail.load()) {
            return nullptr;
        }
        head.store(old_head->next);
        return old_head;
    }
public:
    lock_free_queue():
        head(new node), tail(head.load()) {}
    lock_free_queue(const lock_free_queue& other) =delete;
    lock_free_queue& operator=(const lock_free_queue& other) =delete;
    ~lock_free_queue() {
        while (node* const old_head=head.load()) {
            head.store(old_head->next);
            delete old_head;
        }
    }
    std::shared_ptr<T> pop () {

        node* old_head=pop_head();
        if(!old_head) {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> const res(old_head->data);
        delete old_head;
        return res;
    }

    void push(T new_value) {
        std::shared_ptr<T> new_data(std::make_shared<T>(new_value));
        node* p=new node;
        node* const old_tail=tail.load();
        old_tail->data.swap(new_data);
        old_tail->next=p;
        tail.store(p);
    }
};

struct ReaderToInference {
    cv::Mat origin_frame;
    std::chrono::system_clock::time_point capture_time;
    ReaderToInference(cv::Mat& frame, std::chrono::system_clock::time_point time) 
        : origin_frame(frame.clone()), capture_time(time) {};
};

struct InferenceToTrack {
    cv::Mat origin_frame;
    std::chrono::system_clock::time_point capture_time;
    object_detect_result_list od_results;
    InferenceToTrack(cv::Mat& frame, std::chrono::system_clock::time_point time, object_detect_result_list od_results) 
        : origin_frame(frame.clone()), capture_time(time), od_results(od_results) {};
};

struct TrackToOSD {
    cv::Mat processed_frame;
    std::chrono::system_clock::time_point capture_time;
    TrackToOSD(cv::Mat& frame, std::chrono::system_clock::time_point time) 
        : processed_frame(frame.clone()), capture_time(time) {};
};

#endif 
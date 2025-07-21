#ifndef PIPELINE_H
#define PIPELINE_H

#include <queue>              
#include <mutex>              
#include <condition_variable> 
#include <memory>             
#include <utility>            

#include <pipeline.h>
#include <yolov8.h>
#include <reader.h>
#include <osd.h>

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
};

void img_inference(const char* model_path, const char* input);
void sync(const char* model_path, const char* input);
void async(const char* model_path, const char* input);
void read(Reader& reader);
void run(YoloV8& inference);
void display(OSD& osd);
#endif // THREADSAFE_QUEUE_H


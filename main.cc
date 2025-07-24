#include "pipeline.h"
#include <string>

int main() {
    const char* model_path = "../model/yolov8.rknn";

    const char* img_path = "../busstop.jpg";
    // std::string input =  "rtsp://103.147.186.175:8554/9L02DA3PAJF8FF7";
    // std::string input = "rtsp://user03:abcd1234@113.177.126.84:8202";
    std::string input = "/dev/video0";
    // img_inference(model_path, img_path);
    // sync(model_path, input);
    async(model_path, input);
}
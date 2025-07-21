#include "pipeline.h"


int main() {
    const char* model_path = "../model/yolov8.rknn";
    const char* img_path = "../busstop.jpg";
    // const char* input = "/dev/video0";
    const char* input = "rtsp://103.147.186.175:8554/9L02DA3PAJF8FF7";

    // img_inference(model_path, img_path);
    // sync(model_path, input);
    async(model_path, input);
}
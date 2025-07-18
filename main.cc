#include <iostream>
#include <string>

#include <chrono>

#include <opencv2/opencv.hpp>

#include "yolov8.h"


int main() {
    const char model_path[] = "../model/yolov8.rknn";
    const char img_path[] = "../busstop.jpg";
    std::string input = "/dev/video0";
    // std::string input = " ";
    cv::VideoCapture cap(input);

    std::chrono::steady_clock::time_point Tbegin, Tend;

    YoloV8 detector;
    cv::Mat orig_img;

    int ret;
    ret = detector.init(model_path);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    // // Input
    // orig_img=cv::imread(img_path, 1);
    // if (orig_img.empty())
    // {
    //     printf("Error grabbing img\n");
    //     goto out;
    // }
    
    // //Inference
    // ret = detector.run(orig_img);
    // if (ret != 0)
    // {
    //     printf("inference fail\n");
    //     goto out;
    // }

    // //Output
    // ret = detector.draw(orig_img);
    // if (ret != 0)
    // {
    //     printf("draw fail\n");
    //     goto out;
    // }
    // imwrite("../out.jpg", orig_img);

    while (true) {
        Tbegin = std::chrono::steady_clock::now();
        cap.read(orig_img);
        if (orig_img.empty()) {
            break;
        }
        else {
            ret = detector.run(orig_img);
            if (ret != 0)
            {
                printf("inference fail\n");
                break;
            }
            ret = detector.draw(orig_img);
            if (ret != 0)
            {
                printf("draw fail\n");
                goto out;
            }

            Tend = std::chrono::steady_clock::now();
            double duration_s = std::chrono::duration_cast<std::chrono::microseconds>(Tend - Tbegin).count() / 1000000.0;
            double fps = 1.0 / duration_s;
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << fps; // Format fps to 2 decimal places
            std::string fps_text = "FPS: " + ss.str(); 
            cv::putText(orig_img, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            cv::imshow("sync",orig_img);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    }
    cap.release();
    cv::destroyAllWindows();
    goto out;
out:
    detector.release();
    return 0;
}
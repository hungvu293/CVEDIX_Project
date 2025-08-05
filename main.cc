#include "manager.h"
#include <string>

int main() {
    // std::string input0 =  "rtsp://103.147.186.175:8554/9L02DA3PAJ39B2F";
    std::string input1 = "rtsp://user03:abcd1234@113.177.126.84:8202";
    // std::string input2 = "rtsp://admin:Admin123456@192.168.1.200:8554/cam/realmonitor?channel=1&subtype=0";
    std::vector<std::string> input = {input1, input1};
    Pipeline pipeline;
    int ret = pipeline.initialize(input);
    if (ret != 0) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return -1;
    }
    pipeline.start();
    
    std::string line;
    std::cout << "Press Enter to stop the pipeline..." << std::endl;
    std::getline(std::cin, line);
    pipeline.stop();

}

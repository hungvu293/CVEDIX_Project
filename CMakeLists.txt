cmake_minimum_required(VERSION 3.10)

project(yolov8)

find_package(OpenCV REQUIRED)   

set(SOURCES source/yolov8.cc)

set(RKNN_LIBS ${OpenCV_LIBS} /usr/local/lib/librknnrt.so)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(yolov8 ${SOURCES})

target_link_libraries(yolov8 ${RKNN_LIBS})



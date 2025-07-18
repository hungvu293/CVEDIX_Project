cmake_minimum_required(VERSION 3.10)

project(inference)

find_package(OpenCV REQUIRED)   

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

set(YOLOV8_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source/yolov8.cc)
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)

set(RKNN_RUNTIME_LIB /usr/local/lib/librknnrt.so)

add_library(yolov8_lib STATIC ${YOLOV8_SOURCE})

target_link_libraries(yolov8_lib
    ${OpenCV_LIBS}
    ${RKNN_RUNTIME_LIB}
)

add_executable(inference ${MAIN_SOURCE})

target_link_libraries(inference
    yolov8_lib 
)






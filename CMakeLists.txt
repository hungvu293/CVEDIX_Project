cmake_minimum_required(VERSION 3.10)

project(Camera)

find_package(OpenCV REQUIRED)

add_executable(camera camera.cpp)

target_link_libraries(camera ${OpenCV_LIBS})
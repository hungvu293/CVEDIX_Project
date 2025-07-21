cmake_minimum_required(VERSION 3.10)
project(inference)
find_package(OpenCV REQUIRED)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")
set(RKNN_RUNTIME_LIB /usr/local/lib/librknnrt.so)
add_library(project_core_lib STATIC ${ALL_SOURCE_FILES})
target_link_libraries(project_core_lib
    ${OpenCV_LIBS}
    ${RKNN_RUNTIME_LIB}
)
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference ${MAIN_SOURCE})
target_link_libraries(inference
    project_core_lib
)
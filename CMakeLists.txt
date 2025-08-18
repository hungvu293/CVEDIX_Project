cmake_minimum_required(VERSION 3.10)
project(inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

find_package(PkgConfig REQUIRED)

# Find required packages
pkg_check_modules(RGA REQUIRED librga)
message(STATUS "RGA include directories: ${RGA_INCLUDE_DIRS}")
message(STATUS "RGA libraries: ${RGA_LIBRARIES}")
pkg_check_modules(MPP REQUIRED rockchip_mpp)

# set(ENV{PKG_CONFIG_PATH} "/usr/lib/pkgconfig")
# Find FFmpeg
# pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libswscale libavutil libavfilter libavdevice libpostproc libswresample)

find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PahoMqttCpp REQUIRED)
find_package(Crow)

# set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/oc_sort/deploy/OCSort/cpp)
set (BYTE_TRACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/byte_track/deploy/ncnn/cpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    # ${OC_SORT_ROOT}/include
    # ${OC_SORT_ROOT}/src
    ${BYTE_TRACK_ROOT}/include
    ${Eigen3_INCLUDE_DIR}
    ${RGA_INCLUDE_DIRS}
    ${MPP_INCLUDE_DIRS}
    /home/orangepi/CVEDIX_Project/3rdparty/rknpu2/include
    /home/orangepi
)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")

# file(GLOB OC_SORT_SOURCE_FILES "${OC_SORT_ROOT}/src/*.cpp")
set(BYTE_TRACK_SOURCE_FILES
    "${BYTE_TRACK_ROOT}/src/BYTETracker.cpp"
    "${BYTE_TRACK_ROOT}/src/kalmanFilter.cpp"
    "${BYTE_TRACK_ROOT}/src/lapjv.cpp"
    "${BYTE_TRACK_ROOT}/src/STrack.cpp"
    "${BYTE_TRACK_ROOT}/src/utils.cpp"
)
add_library(project_core_lib STATIC
    ${ALL_SOURCE_FILES}
    # ${OC_SORT_SOURCE_FILES}
    ${BYTE_TRACK_SOURCE_FILES}
)

target_link_libraries(project_core_lib
    Eigen3::Eigen
    ${OpenCV_LIBS}
    /usr/lib/libavcodec.so
    /usr/lib/libavformat.so
    /usr/lib/libavutil.so
    /usr/lib/libavfilter.so
    /usr/lib/libavdevice.so
    /usr/lib/libswscale.so
    /usr/lib/libswresample.so
    /usr/lib/libpostproc.so
    ${RGA_LIBRARIES}
    ${MPP_LIBRARIES}
    Crow::Crow
    paho-mqttpp3
    paho-mqtt3as
    /home/orangepi/CVEDIX_Project/3rdparty/rknpu2/Linux/aarch64/librknnrt.so
)

# Create executable and link libraries
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference
    ${MAIN_SOURCE}
)

target_link_libraries(inference
    project_core_lib
)

# Update Git submodules
find_package(Git QUIET)
if(GIT_FOUND)
    add_custom_target(submodule_update ALL
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(inference submodule_update)
    add_dependencies(project_core_lib submodule_update)
endif()
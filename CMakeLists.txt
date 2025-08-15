set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.10)
project(inference CXX)

set(CUSTOM_INSTALL_PREFIX "$ENV{HOME}/dev/ffmpeg_build")

find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PkgConfig QUIET)
find_package(PahoMqttCpp REQUIRED)
find_package(Crow)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(RGA REQUIRED librga)
    pkg_check_modules(MPP REQUIRED rockchip_mpp)
    pkg_check_modules(VPU REQUIRED rockchip_vpu)
    set(ENV{PKG_CONFIG_PATH} "${CUSTOM_INSTALL_PREFIX}/lib/pkgconfig")
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libswscale libavutil libavfilter libavdevice libpostproc libswresample)
    # pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
endif()

set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/oc_sort/deploy/OCSort/cpp)

set(RKNN_RUNTIME_LIB ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OC_SORT_ROOT}/include
    ${OC_SORT_ROOT}/src
    ${Eigen3_INCLUDE_DIR}
    ${FFMPEG_INCLUDE_DIRS}
    # ${GSTREAMER_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
    ${MPP_INCLUDE_DIRS}
    ${VPU_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include
)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")

file(GLOB OC_SORT_SOURCE_FILES
    "${OC_SORT_ROOT}/src/*.cpp"
)

add_library(project_core_lib STATIC
    ${ALL_SOURCE_FILES}
    ${OC_SORT_SOURCE_FILES}
)

target_link_libraries(project_core_lib
    Eigen3::Eigen
)

set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference
    ${MAIN_SOURCE}
)

target_link_libraries(inference
    project_core_lib      
    ${OpenCV_LIBS}        
    ${FFMPEG_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${RKNN_RUNTIME_LIB}   
    ${RGA_LIBRARIES}
    ${MPP_LIBRARIES}
    ${VPU_LIBRARIES}
    Crow::Crow
    paho-mqttpp3
    paho-mqtt3as
)

find_package(Git QUIET)
if(GIT_FOUND)
    add_custom_target(submodule_update ALL
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(inference submodule_update)
    add_dependencies(project_core_lib submodule_update)
endif()
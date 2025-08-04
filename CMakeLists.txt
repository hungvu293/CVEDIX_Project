set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.10)
project(inference CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFMPEG REQUIRED opencv4 libavcodec libavformat libavutil libswscale)
    if(NOT FFMPEG_FOUND)
        message(FATAL_ERROR "FFmpeg not found. Please ensure pkg-config can find it.")
    endif()
else()
    message(FATAL_ERROR "pkg-config not found. Please install pkg-config to find FFmpeg.")
endif()

set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/oc_sort/deploy/OCSort/cpp)

set(RKNN_RUNTIME_LIB ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so)
set(RGA_RUNTIME_LIB_SO ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/libs/Linux/gcc-aarch64/librga.so)
set(RGA_RUNTIME_LIB_A ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/libs/Linux/gcc-aarch64/librga.a) 

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OC_SORT_ROOT}/include
    ${OC_SORT_ROOT}/src
    ${OC_SORT_ROOT}/detector
    ${Eigen3_INCLUDE_DIR}
    ${FFMPEG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include
    /usr/include/libdrm
)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")

file(GLOB OC_SORT_SOURCE_FILES
    "${OC_SORT_ROOT}/src/*.cpp"
    "${OC_SORT_ROOT}/detector/*.cpp"
)

add_library(project_core_lib STATIC
    ${ALL_SOURCE_FILES}
    ${OC_SORT_SOURCE_FILES}
)

target_link_libraries(project_core_lib
    Eigen3::Eigen
)

set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference ${MAIN_SOURCE})

target_link_libraries(inference
    project_core_lib      
    ${OpenCV_LIBS}        
    ${FFMPEG_LIBRARIES}   
    ${RKNN_RUNTIME_LIB}   
    ${RGA_RUNTIME_LIB_SO} 
    ${RGA_RUNTIME_LIB_A}  
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
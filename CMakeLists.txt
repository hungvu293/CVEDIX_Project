# --- Cài đặt cơ bản cho dự án ---
cmake_minimum_required(VERSION 3.10)
project(inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")


# --- Tìm kiếm các gói phụ thuộc bằng PkgConfig (Cách chuẩn) ---
find_package(PkgConfig REQUIRED)

# Sử dụng pkg-config để tìm FFmpeg. Đây là cách chính xác vì bạn đã cài vào /usr
# Nó sẽ tự động điền các biến FFMPEG_INCLUDE_DIRS và FFMPEG_LIBRARIES
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libswscale libavutil libavfilter libavdevice libpostproc libswresample)

# Tìm các thư viện phần cứng Rockchip
pkg_check_modules(RGA REQUIRED librga)
pkg_check_modules(MPP REQUIRED rockchip_mpp)
# pkg_check_modules(VPU REQUIRED rockchip_vpu) # VPU có thể không cần thiết cho việc decode

# Tìm các gói khác
find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PahoMqttCpp REQUIRED)
find_package(Crow)


# --- Khai báo các đường dẫn Include ---
set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/oc_sort/deploy/OCSort/cpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OC_SORT_ROOT}/include
    ${OC_SORT_ROOT}/src
    ${Eigen3_INCLUDE_DIR}
    ${FFMPEG_INCLUDE_DIRS} # Biến này được tạo tự động bởi pkg_check_modules
    ${RGA_INCLUDE_DIRS}
    ${MPP_INCLUDE_DIRS}
    # ${VPU_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include
)


# --- Định nghĩa các file Source và tạo thư viện ---
set(RKNN_RUNTIME_LIB ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so)

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


# --- Tạo file thực thi và liên kết thư viện ---
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference
    ${MAIN_SOURCE}
)

target_link_libraries(inference
    project_core_lib
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES} # Biến này được tạo tự động bởi pkg_check_modules
    ${RKNN_RUNTIME_LIB}
    ${RGA_LIBRARIES}
    ${MPP_LIBRARIES}
    # ${VPU_LIBRARIES}
    Crow::Crow
    paho-mqttpp3
    paho-mqtt3as
)

# QUAN TRỌNG: Thêm RPATH để đảm bảo file thực thi tìm đúng thư viện khi chạy
# Điều này giúp tránh xung đột với các phiên bản thư viện khác trên hệ thống.
# target_link_options(inference PRIVATE "-Wl,-rpath,${FFMPEG_LIBRARY_DIRS}")
set_target_properties(inference PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64"
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64"
)


# --- Cập nhật Git Submodule ---
find_package(Git QUIET)
if(GIT_FOUND)
    add_custom_target(submodule_update ALL
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(inference submodule_update)
    add_dependencies(project_core_lib submodule_update)
endif()

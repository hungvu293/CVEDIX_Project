cmake_minimum_required(VERSION 3.10)
project(inference CXX)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PkgConfig QUIET) # Required for FFmpeg

if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFMPEG REQUIRED opencv4 libavcodec libavformat libavutil libswscale)
    if(NOT FFMPEG_FOUND)
        message(FATAL_ERROR "FFmpeg not found. Please ensure pkg-config can find it.")
    endif()
else()
    message(FATAL_ERROR "pkg-config not found. Please install pkg-config to find FFmpeg.")
endif()

# Set OC_SORT_ROOT
set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/oc_sort/deploy/OCSort/cpp)

# Define paths for RKNN and RGA runtime libraries (nên dùng full path hoặc add_library nếu là static)
# RKNN và RGA thường là shared libraries, nên để nguyên đường dẫn full .so
set(RKNN_RUNTIME_LIB ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so)
set(RGA_RUNTIME_LIB_SO ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/libs/Linux/gcc-aarch64/librga.so)
set(RGA_RUNTIME_LIB_A ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/libs/Linux/gcc-aarch64/librga.a) # Nếu bạn thực sự cần cái này, nó sẽ được link tĩnh

# Include directories
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
)

# Source files
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")

file(GLOB OC_SORT_SOURCE_FILES
    "${OC_SORT_ROOT}/src/*.cpp"
    "${OC_SORT_ROOT}/detector/*.cpp"
)

# Add project_core_lib static library
# CHÚ Ý: Các file mã nguồn của OCSort phải được biên dịch cùng với project_core_lib
# nếu track.cc gọi ocsort
add_library(project_core_lib STATIC
    ${ALL_SOURCE_FILES}
    ${OC_SORT_SOURCE_FILES} # <-- OCSort source files được thêm vào đây
)

# KHÔNG LIÊN KẾT CÁC THƯ VIỆN ĐỘNG VÀO THƯ VIỆN TĨNH!
# Các phụ thuộc của project_core_lib được truyền qua INTERFACE/PUBLIC nếu cần,
# nhưng với STATIC lib, tốt nhất là link thẳng vào executable.
# Bỏ qua target_link_libraries cho project_core_lib nếu nó chỉ là một gói mã nguồn.
# Nếu project_core_lib thực sự CẦN các thư viện này để biên dịch nó,
# thì bạn có thể dùng target_link_libraries(project_core_lib PRIVATE ...)
# Nhưng các thư viện đó phải là các thư viện mà project_core_lib đang GỌI.
# Trong trường hợp này, project_core_lib đang GỌI OCSort, mà OCSort GỌI Eigen.
# Nên project_core_lib cần biết Eigen.
target_link_libraries(project_core_lib
    Eigen3::Eigen # project_core_lib (thông qua ocsort) sử dụng Eigen
)


# Add executable
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference ${MAIN_SOURCE})

# Liên kết project_core_lib VÀ TẤT CẢ CÁC PHỤ THUỘC CỦA NÓ (và inference) vào executable cuối cùng
target_link_libraries(inference
    project_core_lib      # Liên kết với thư viện tĩnh của bạn
    ${OpenCV_LIBS}        # OpenCV
    ${FFMPEG_LIBRARIES}   # FFmpeg
    ${RKNN_RUNTIME_LIB}   # RKNN Runtime
    ${RGA_RUNTIME_LIB_SO} # RGA Runtime (shared)
    ${RGA_RUNTIME_LIB_A}  # RGA Runtime (static - nếu cần, thường chỉ .so)
    # Eigen3::Eigen      # Đã được link vào project_core_lib, không cần ở đây nữa nếu project_core_lib là PRIVATE link
)

# Git submodule update
find_package(Git QUIET) # Check if Git is available
if(GIT_FOUND)
    add_custom_target(submodule_update ALL
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(inference submodule_update)
    add_dependencies(project_core_lib submodule_update) # Tùy chọn, có thể bỏ qua
endif()
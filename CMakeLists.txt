cmake_minimum_required(VERSION 3.10)
project(inference CXX) # Specify CXX for C++ project

# --- Find required packages ---
find_package(OpenCV REQUIRED)

# Find Eigen3
# IMPORTANT: Ensure Eigen3 is installed on your system.
# Use 'sudo apt install libeigen3-dev' on Ubuntu/Debian.
# If installed manually, you might need to specify Eigen3_DIR like:
# set(Eigen3_DIR "/path/to/your/eigen-build-folder") # e.g., /home/orangepi/eigen-3.4.0/build
find_package(Eigen3 CONFIG REQUIRED) # Use CONFIG for modern Eigen installations

# --- Define OC_SORT Submodule paths ---
# Base path to the OC_SORT C++ implementation within your submodule
set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/oc_sort/deploy/OCSort/cpp)

# --- Include Directories ---
# Add all necessary include paths for your project and its dependencies
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include        # Your project's own headers
    ${OpenCV_INCLUDE_DIRS}                     # OpenCV headers
    ${OC_SORT_ROOT}/include                    # OC_SORT's main headers (e.g., Association.hpp, KalmanBoxTracker.hpp)
    ${OC_SORT_ROOT}/src                        # <--- THIS IS THE CRUCIAL ADDITION for kalmanfilter.hpp and other internal includes
    ${OC_SORT_ROOT}/detector                   # If detector has headers (e.g., inference.h)
    ${Eigen3_INCLUDE_DIR}                      # Eigen headers (important for "Eigen/Dense")
)

# --- Source Files ---
# Your project's core source files
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")

# OC_SORT submodule source files
file(GLOB OC_SORT_SOURCE_FILES
    "${OC_SORT_ROOT}/src/*.cpp"
    "${OC_SORT_ROOT}/detector/*.cpp" # Include if there are .cpp files in detector
)

# --- Create Libraries ---
set(RKNN_RUNTIME_LIB /usr/local/lib/librknnrt.so)

add_library(project_core_lib STATIC
    ${ALL_SOURCE_FILES}
    ${OC_SORT_SOURCE_FILES} # Include OC_SORT's source files here
)

# --- Link Libraries for project_core_lib ---
target_link_libraries(project_core_lib
    ${OpenCV_LIBS}
    ${RKNN_RUNTIME_LIB}
    Eigen3::Eigen # Link with Eigen3
)

# --- Create Executable ---
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference ${MAIN_SOURCE})

# --- Link Libraries for inference executable ---
target_link_libraries(inference
    project_core_lib
)

# --- Submodule Update Hook (Crucial for CI/CD and multi-user projects) ---
find_package(Git QUIET) # Check if Git is available
if(GIT_FOUND)
    add_custom_target(submodule_update ALL
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # Ensure the executable and library compilation depends on the submodule being updated
    add_dependencies(inference submodule_update)
    add_dependencies(project_core_lib submodule_update)
endif()
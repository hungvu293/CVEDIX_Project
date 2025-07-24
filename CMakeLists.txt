cmake_minimum_required(VERSION 3.10)
project(inference CXX) 

find_package(OpenCV REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)

set(OC_SORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/oc_sort/deploy/OCSort/cpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include        
    ${OpenCV_INCLUDE_DIRS}                     
    ${OC_SORT_ROOT}/include                    
    ${OC_SORT_ROOT}/src                        
    ${OC_SORT_ROOT}/detector                   
    ${Eigen3_INCLUDE_DIR}                      
)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB ALL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cc")

file(GLOB OC_SORT_SOURCE_FILES
    "${OC_SORT_ROOT}/src/*.cpp"
    "${OC_SORT_ROOT}/detector/*.cpp"
)

set(RKNN_RUNTIME_LIB /usr/local/lib/librknnrt.so)

add_library(project_core_lib STATIC
    ${ALL_SOURCE_FILES}
    ${OC_SORT_SOURCE_FILES} 
)

target_link_libraries(project_core_lib
    ${OpenCV_LIBS}
    ${RKNN_RUNTIME_LIB}
    Eigen3::Eigen
)

set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
add_executable(inference ${MAIN_SOURCE})

target_link_libraries(inference
    project_core_lib
)

find_package(Git QUIET) # Check if Git is available
if(GIT_FOUND)
    add_custom_target(submodule_update ALL
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(inference submodule_update)
    add_dependencies(project_core_lib submodule_update)
endif()
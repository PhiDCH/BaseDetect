cmake_minimum_required(VERSION 2.8.3)
project(yolov5)

## Compile as C++11, supported in ROS Kinetic and newer
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


## System dependencies are found with CMake's conventions
find_package(CUDA REQUIRED)
find_package(OpenCV)

find_library(BASE_DETECT libbase_detect.so)
include_directories(/usr/local/include)


include_directories(include)
link_directories(include)

include_directories(../base_detect/include)
link_directories(../base_detect/include)

include_directories(${OpenCV_INCLUDE_DIRS})


# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt and cudnn
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)


cuda_add_executable(yolov5 src/yolov5.cpp src/preprocess.cu)
target_link_libraries(yolov5
  # PUBLIC ${BASE_DETECT}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
)


if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


# add_library(yolov5 SHARED src/yolov5.cpp)

# install(TARGETS yolov5 DESTINATION /usr/lib)
# install(
#   FILES include/yolov5.h
#   DESTINATION include
# ) 
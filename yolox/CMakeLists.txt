cmake_minimum_required(VERSION 2.8.3)
project(yolox)

## Compile as C++11, supported in ROS Kinetic and newer
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -fopenmp")


## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

find_package(CUDA REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(
    ${Boost_INCLUDE_DIRS}
)

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

add_library(yolox src/yolox.cpp)

file(GLOB My_Source_Files src/*.cpp)
# add_executable(base_detect ${My_Source_Files})
add_executable(testYoloX src/test.cpp)
target_link_libraries(testYoloX
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
  Eigen3::Eigen
)


add_definitions(-O2 -pthread)


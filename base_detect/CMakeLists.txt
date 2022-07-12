cmake_minimum_required(VERSION 2.8.3)
project(base_detect)

## Compile as C++11, supported in ROS Kinetic and newer
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


## System dependencies are found with CMake's conventions
find_package(CUDA REQUIRED)
find_package(OpenCV)
# find_package(Eigen3 3.4 REQUIRED NO_MODULE)


include_directories(include)
link_directories(include)

include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt and cudnn
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)


# add_executable(base_detect src/base_detect.cpp)
# target_link_libraries(base_detect
#   ${OpenCV_LIBRARIES}
#   cudart
#   nvinfer
# )

# add_definitions(-O2 -pthread)

add_library(base_detect SHARED src/base_detect.cpp)

install(TARGETS base_detect DESTINATION /usr/lib)
install(
  FILES include/base_detect.h include/logging.h
  DESTINATION include/base_detect
)



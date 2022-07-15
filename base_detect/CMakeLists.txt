cmake_minimum_required(VERSION 2.8.3)
project(base_detect)

## Compile as C++11, supported in ROS Kinetic and newer
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


## System dependencies are found with CMake's conventions

# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# tensorrt and cudnn
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu/)
  link_directories(/usr/lib/aarch64-linux-gnu/)
else()
  message("embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu/)
  link_directories(/usr/lib/x86_64-linux-gnu/)
endif()

#local include
include_directories(include)
link_directories(include)


# add_executable(base_detect src/base_detect.cpp)
# target_link_libraries(base_detect
#   ${OpenCV_LIBRARIES}
#   cudart
#   nvinfer
# )


add_library(base_detect SHARED src/base_detect.cpp)

install(TARGETS base_detect DESTINATION /usr/lib)
install(
  FILES include/base_detect.h include/logging.h
  DESTINATION include
)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

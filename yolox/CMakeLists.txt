cmake_minimum_required(VERSION 2.8.3)
project(yolox)

## Compile as C++11, supported in ROS Kinetic and newer
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -fopenmp")


## System dependencies are found with CMake's conventions

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# opencv
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



# base_detect
find_library(BASE_DETECT libbase_detect.so)
include_directories(/usr/local/include)



# add_executable(yolox yolox.cpp)
# target_link_libraries(yolox
#   PUBLIC ${BASE_DETECT}
#   ${OpenCV_LIBRARIES}
#   cudart
#   nvinfer
#   # Eigen3::Eigen
# )


add_library(yolox SHARED yolox.cpp)
install(TARGETS yolox DESTINATION /usr/lib)
install(
  FILES yolox.h
  DESTINATION include
) 

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
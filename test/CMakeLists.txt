cmake_minimum_required(VERSION 2.8.3)
project(testModule)

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

# eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

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



# local lib
find_library(BASE_DETECT libbase_detect.so)
find_library(YOLOX libyolox.so)
find_library(BYTETRACK libbytetrack.so)
find_library(YOLOV5 libyolov5.so)
find_library(YOLOV6 libyolov6.so)
include_directories(/usr/local/include)




add_executable(test_base test_base.cpp)
target_link_libraries(test_base
  PUBLIC ${BASE_DETECT}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
)


add_executable(det_yolox det_yolox.cpp)
target_link_libraries(det_yolox
  PUBLIC ${YOLOX}
  PUBLIC ${BASE_DETECT}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
)


add_executable(track_bytetrack_yolox track_bytetrack_yolox.cpp)
target_link_libraries(track_bytetrack_yolox
  PUBLIC ${BASE_DETECT}
  PUBLIC ${YOLOX}
  PUBLIC ${BYTETRACK}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
  Eigen3::Eigen
)


add_executable(det_yolov5 det_yolov5.cpp)
target_link_libraries(det_yolov5
  PUBLIC ${YOLOV5}
  PUBLIC ${BASE_DETECT}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
)

add_executable(track_bytetrack_yolov5 track_bytetrack_yolov5.cpp)
target_link_libraries(track_bytetrack_yolov5
  PUBLIC ${BASE_DETECT}
  PUBLIC ${YOLOV5}
  PUBLIC ${BYTETRACK}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
  Eigen3::Eigen
)

add_executable(det_yolov6 det_yolov6.cpp)
target_link_libraries(det_yolov6
  PUBLIC ${YOLOV6}
  PUBLIC ${BASE_DETECT}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
)

add_executable(track_bytetrack_yolov6 track_bytetrack_yolov6.cpp)
target_link_libraries(track_bytetrack_yolov6
  PUBLIC ${BASE_DETECT}
  PUBLIC ${YOLOV6}
  PUBLIC ${BYTETRACK}
  ${OpenCV_LIBRARIES}
  cudart
  nvinfer
  Eigen3::Eigen
)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


cmake_minimum_required(VERSION 2.8.3)
project(base_detect)

## Compile as C++11, supported in ROS Kinetic and newer
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -fopenmp")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    actionlib
    actionlib_msgs
    message_generation
)
## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

find_package(CUDA REQUIRED)
find_package(OpenCV)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


catkin_package()

# ## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   HumanBox.msg
#   HumanMatch.msg
# )

# ## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   EnableHumanDetect.srv
#   EnableHumanMatching.srv
# )

# ## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   sensor_msgs
#   actionlib_msgs
# )

# catkin_package(
#   LIBRARIES human_tracking_navigation
#   CATKIN_DEPENDS roscpp rospy std_msgs actionlib actionlib_msgs message_runtime sensor_msgs
# )


include_directories(
    ${catkin_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/usr/include/eigen3)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt and cudnn
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

file(GLOB My_Source_Files src/*.cpp)
# add_executable(base_detect ${My_Source_Files})
add_executable(base_detect src/base_detect.cpp)
target_link_libraries(base_detect
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  cudart
  nvinfer
  Eigen3::Eigen
)

add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})
# add_dependencies(talker beginner_tutorials_generate_messages_cpp)

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})
# add_dependencies(listener beginner_tutorials_generate_messages_cpp)


add_definitions(-O2 -pthread)


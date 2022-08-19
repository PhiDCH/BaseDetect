cmake_minimum_required(VERSION 2.8.3)
project(bytetrack)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -fopenmp")


## System dependencies are found with CMake's conventions
find_package(OpenCV)
find_package(Eigen3 REQUIRED NO_MODULE)


include_directories(include)
link_directories(include)

include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB MY_SRC src/*.cpp)
file(GLOB MY_HEADER include/*.h)


# add_executable(bytetrack ${MY_SRC})
# target_link_libraries(bytetrack
#   ${OpenCV_LIBRARIES}
#   Eigen3::Eigen
# )

add_definitions(-O2 -pthread)

add_library(bytetrack SHARED ${MY_SRC})

install(TARGETS bytetrack DESTINATION /usr/lib)
install(
  FILES ${MY_HEADER}
  DESTINATION include
)


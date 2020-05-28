cmake_minimum_required(VERSION 2.8)
project(direct_vo)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -std=c++17")

add_executable(direct_vo direct_vo.cpp )
target_link_libraries(direct_vo Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES})

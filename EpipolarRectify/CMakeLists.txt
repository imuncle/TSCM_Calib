cmake_minimum_required(VERSION 2.8.3)
project(EpipolarRectify)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
find_package(OpenCV REQUIRED)

include_directories(
  DetectCorner 
  ${OpenCV_INCLUDE_DIRECTORIES}
)

add_executable(${PROJECT_NAME}
  rectify.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

cmake_minimum_required(VERSION 2.8.3)
project(TSCM_Calib)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
find_package(OpenCV REQUIRED)
find_package( Ceres REQUIRED)

include_directories(
  DetectCorner 
  ${OpenCV_INCLUDE_DIRECTORIES}
  ${CERES_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
  main.cpp
  TS.cpp
  multi_calib.cpp
  DetectCorner/findCorner.cpp
  DetectCorner/chessboard.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
)

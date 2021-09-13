cmake_minimum_required(VERSION 2.8.3)
project(DSCM_Calib)

add_compile_options(-std=c++11)
find_package(OpenCV 3 REQUIRED)


include_directories(
  DetectCorner 
  ${OpenCV_INCLUDE_DIRECTORIES}
)


add_executable(${PROJECT_NAME}
  main.cpp
  DS.cpp
  DetectCorner/findCorner.cpp
  DetectCorner/chessboard.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)
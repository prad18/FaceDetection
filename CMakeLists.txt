cmake_minimum_required(VERSION 3.15)

project(FaceDetection)

set(OpenCV_DIR D:/opencv/build)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) 
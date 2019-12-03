SET(execName yolov3_detect)
project(${execName})
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++14)
set(OpenCV_DIR "/home/goldenridge/opencv4/opencv-4.0.1/build")

#include_directories("usr/local/include")
#link_directories("usr/local/lib")
find_package( OpenCV REQUIRED )
add_executable( ${execName} yolov3.cpp )
target_link_libraries( ${execName} gflags pthread ${OpenCV_LIBS} )
#install
install( TARGETS ${execName} RUNTIME DESTINATION bin )

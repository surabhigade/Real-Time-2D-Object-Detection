cmake_minimum_required(VERSION 3.10)
project(2DObjectRecognition)

set(CMAKE_CXX_STANDARD 14)
# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
add_executable(2DObjectRecognition main.cpp kmeans.cpp tasks.cpp)

# Link OpenCV libraries
target_link_libraries(2DObjectRecognition ${OpenCV_LIBS})

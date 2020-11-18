cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project( InferenceHandGestures )

find_package(Torch REQUIRED)  

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable(InferenceHandGestures main.cpp handKeyPoints.cpp)

target_link_libraries(
    InferenceHandGestures
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    )

set_property(TARGET InferenceHandGestures PROPERTY CXX_STANDARD 17)

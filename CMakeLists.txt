cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(_NVCC_EXEC nvcc HINTS /usr/local/cuda/bin)
    if(_NVCC_EXEC)
        set(CMAKE_CUDA_COMPILER ${_NVCC_EXEC} CACHE FILEPATH "CUDA compiler" FORCE)
    else()
        message(FATAL_ERROR "nvcc not found")
    endif()
endif()

execute_process(COMMAND uname -m OUTPUT_VARIABLE _HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_HOST_ARCH STREQUAL "aarch64" OR EXISTS "/etc/nv_tegra_release")
    set(_CUDA_ARCH 87)
    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_LINK_DIRS /usr/lib/aarch64-linux-gnu)
else()
    set(_CUDA_ARCH 86)
    set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT root")
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
    set(TENSORRT_LINK_DIRS ${TENSORRT_ROOT}/lib)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${_CUDA_ARCH} CACHE STRING "CUDA architectures" FORCE)

project(yolo26_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(yolo26_trt_node
    src/detect_node.cpp
    src/yolo26.cpp
    src/preprocess.cu
)
add_executable(yolo26_trt_demo
    src/demo.cpp
    src/yolo26.cpp
    src/preprocess.cu
)

foreach(_tgt yolo26_trt_node yolo26_trt_demo)
    target_include_directories(${_tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}
    )
    target_link_directories(${_tgt} PRIVATE ${TENSORRT_LINK_DIRS})
    target_link_libraries(${_tgt} ${OpenCV_LIBS} CUDA::cudart nvinfer)
    target_compile_options(${_tgt} PRIVATE
        -Wno-deprecated-declarations
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )
    set_target_properties(${_tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

ament_target_dependencies(yolo26_trt_node rclcpp sensor_msgs cv_bridge)

install(TARGETS yolo26_trt_node yolo26_trt_demo DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/01_check_jetson.sh scripts/export_engine.sh DESTINATION lib/${PROJECT_NAME})

ament_package()

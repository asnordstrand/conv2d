cmake_minimum_required(VERSION 3.18)

project(conv2d CXX CUDA)

add_executable(${PROJECT_NAME} conv2d.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED TRUE
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas cudnn)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

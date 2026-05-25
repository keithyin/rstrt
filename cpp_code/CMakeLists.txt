cmake_minimum_required(VERSION 3.18)
project(tensorrt_infer)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ✅ 用新方式找 CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT 路径（根据容器情况）
set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)

add_executable(main
    main.cpp
    trt_infer.cpp
)

# ✅ include
target_include_directories(main PRIVATE
    ${TENSORRT_INCLUDE_DIR}
)

# ✅ link 路径
target_link_directories(main PRIVATE
    ${TENSORRT_LIB_DIR}
)

# ✅ link 库（关键）
target_link_libraries(main
    nvinfer
    CUDA::cudart
)
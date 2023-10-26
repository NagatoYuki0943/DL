cmake_minimum_required(VERSION 3.21)
project(test)

set(CMAKE_CXX_STANDARD 14)

# cuda
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Torch
set(Torch_DIR /home/ubuntu/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(${TORCH_INCLUDE_DIRS})  # /home/ubuntu/libtorch/include /home/ubuntu/libtorch/include/torch/csrc/api/include
    message(${TORCH_LIBRARIES})
endif ()
include_directories(${TORCH_INCLUDE_DIRS})

# 链接所有库，不指定cpp文件
link_libraries(${TORCH_LIBRARIES})

# add_executable写在下面
add_executable(main main.cpp)

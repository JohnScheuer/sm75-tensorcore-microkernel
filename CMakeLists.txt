cmake_minimum_required(VERSION 3.18)

project(micro_kernel_generator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------
# Kernel Library
# --------------------------------------------------

file(GLOB KERNEL_SOURCES kernels/*.cu)

add_library(kernels STATIC ${KERNEL_SOURCES})

set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --------------------------------------------------
# Benchmark Executable
# --------------------------------------------------

add_executable(benchmark benchmarks/benchmark.cu)

target_link_libraries(benchmark
    PRIVATE
    kernels
    CUDA::cudart
    CUDA::cublas
)

# Force C++ linker
set_target_properties(benchmark PROPERTIES
    LINKER_LANGUAGE CXX
)
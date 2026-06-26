cmake_minimum_required(VERSION 3.18)

project(micro_kernel_generator LANGUAGES CXX CUDA)

# --------------------------------------------------
# Standards
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------
# Default Tunable Parameters
# (can override via -D in cmake)
# --------------------------------------------------

set(BLOCK_M 64 CACHE STRING "Block tile M dimension")
set(BLOCK_N 64 CACHE STRING "Block tile N dimension")
set(STAGE_K 16 CACHE STRING "K stage depth")
set(WARPS_PER_BLOCK 4 CACHE STRING "Warps per block")
set(ILP_DEPTH 2 CACHE STRING "Accumulator sets (1 or 2)")

# --------------------------------------------------
# Compile Definitions
# --------------------------------------------------

add_compile_definitions(
    BLOCK_M=${BLOCK_M}
    BLOCK_N=${BLOCK_N}
    STAGE_K=${STAGE_K}
    WARPS_PER_BLOCK=${WARPS_PER_BLOCK}
    ILP_DEPTH=${ILP_DEPTH}
)

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
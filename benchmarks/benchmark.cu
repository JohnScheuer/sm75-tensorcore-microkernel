#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                  \
    do {                                                  \
        cudaError_t err = call;                           \
        if (err != cudaSuccess) {                         \
            std::cerr << "CUDA Error: "                   \
                      << cudaGetErrorString(err)          \
                      << std::endl;                       \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

#define CHECK_CUBLAS(call)                                \
    do {                                                  \
        cublasStatus_t status = call;                     \
        if (status != CUBLAS_STATUS_SUCCESS) {            \
            std::cerr << "cuBLAS Error\n";                \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

__global__ void gemm_ptx_hybrid(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K);

double benchmark_ptx(half* dA, half* dB, float* dC,
                     int M, int N, int K)
{
    dim3 block(128);
    dim3 grid((N + 63) / 64, (M + 63) / 64);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iters = 10;

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; i++)
        gemm_ptx_hybrid<<<grid, block>>>(dA, dB, dC, M, N, K);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double flops = 2.0 * (double)M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    return gflops;
}

double benchmark_cublas(half* dA, half* dB, float* dC,
                        int M, int N, int K)
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta  = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iters = 10;

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; i++)
    {
        CHECK_CUBLAS(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, CUDA_R_16F, N,
                dA, CUDA_R_16F, K,
                &beta,
                dC, CUDA_R_32F, N,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            )
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double flops = 2.0 * (double)M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    cublasDestroy(handle);

    return gflops;
}

int main()
{
    std::vector<int> sizes = {2048, 4096, 8192};

    const double peak_tflops = 29000.0; // RTX 2070 FP16 Tensor Core peak

    for (int size : sizes)
    {
        int M = size;
        int N = size;
        int K = size;

        size_t sizeA = (size_t)M * K;
        size_t sizeB = (size_t)K * N;
        size_t sizeC = (size_t)M * N;

        half* dA;
        half* dB;
        float* dC;

        CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

        CHECK_CUDA(cudaMemset(dA, 0, sizeA * sizeof(half)));
        CHECK_CUDA(cudaMemset(dB, 0, sizeB * sizeof(half)));
        CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

        double ptx_perf = benchmark_ptx(dA, dB, dC, M, N, K);
        double cublas_perf = benchmark_cublas(dA, dB, dC, M, N, K);

        double efficiency = (ptx_perf / peak_tflops) * 100.0;

        std::cout << "---------------------------------------------\n";
        std::cout << "Size: " << size << " x " << size << "\n";
        std::cout << "PTX Hybrid GEMM: " << ptx_perf << " GFLOPs/s\n";
        std::cout << "cuBLAS:           " << cublas_perf << " GFLOPs/s\n";
        std::cout << "Efficiency vs 29 TFLOPs peak: "
                  << efficiency << " %\n";
        std::cout << "---------------------------------------------\n\n";

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    return 0;
}
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template<int BM, int BN, int BK>
__global__ void gemm_shared_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * BM + ty;
    const int col = blockIdx.x * BN + tx;

    float accum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Load A tile
        if (row < M && (k0 + tx) < K)
            As[ty][tx] = A[row * K + k0 + tx];
        else
            As[ty][tx] = __float2half(0.0f);

        // Load B tile
        if ((k0 + ty) < K && col < N)
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        else
            Bs[ty][tx] = __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++)
        {
            accum += __half2float(As[ty][k]) *
                     __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = accum;
}

// Launcher wrapper
void launch_gemm_shared_fp16(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K)
{
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 16;

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    gemm_shared_fp16_kernel<BM, BN, BK>
        <<<grid, block>>>(A, B, C, M, N, K);
}
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef BLOCK_M
#define BLOCK_M 64
#endif

#ifndef BLOCK_N
#define BLOCK_N 64
#endif

#ifndef STAGE_K
#define STAGE_K 16
#endif

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

#ifndef ILP_DEPTH
#define ILP_DEPTH 2
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ---- PTX Compute Core ----
__device__ inline void wmma_core(float* c, unsigned* a, unsigned* b)
{
    asm volatile(
        "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
}

// ---- GEMM Kernel ----
__global__ void gemm_ptx_hybrid(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[BLOCK_M][WMMA_K];
    __shared__ half Bs[WMMA_K][BLOCK_N];

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warpRow = warpId / (BLOCK_N / 32);
    int warpCol = warpId % (BLOCK_N / 32);

    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    float c[8] = {0};
#if ILP_DEPTH == 2
    float d[8] = {0};
#endif

    for (int k0 = 0; k0 < K; k0 += STAGE_K)
    {
        // ---- Load A tile ----
        int loadRow = threadIdx.x / WMMA_K;
        int loadCol = threadIdx.x % WMMA_K;

        if (loadRow < BLOCK_M)
        {
            int gRow = blockRow + loadRow;
            int gCol = k0 + loadCol;
            As[loadRow][loadCol] =
                (gRow < M && gCol < K) ?
                A[gRow*K + gCol] : __float2half(0.0f);
        }

        // ---- Load B tile ----
        if (loadRow < WMMA_K)
        {
            int gRow = k0 + loadRow;
            int gCol = blockCol + loadCol;
            Bs[loadRow][loadCol] =
                (gRow < K && gCol < N) ?
                B[gRow*N + gCol] : __float2half(0.0f);
        }

        __syncthreads();

        unsigned a[8];
        unsigned b[8];

        int aBase = warpRow * 32;
        int bBase = warpCol * 32;

        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            a[i] = *((unsigned*)&As[aBase + (i%2)*16][0]);
            b[i] = *((unsigned*)&Bs[0][bBase + (i%2)*16]);
        }

        // ---- Compute ----
        wmma_core(c, a, b);

#if ILP_DEPTH == 2
        wmma_core(d, a, b);
#endif

        __syncthreads();
    }

    // ---- Store results ----
    int cRow = blockRow + warpRow * 32;
    int cCol = blockCol + warpCol * 32;

    if (laneId == 0)
    {
        for (int i = 0; i < 8; i++)
        {
            int rowOffset = (i / 2) * 16;
            int colOffset = (i % 2) * 16;

            if (cRow + rowOffset < M && cCol + colOffset < N)
            {
                C[(cRow + rowOffset) * N + (cCol + colOffset)] =
                    c[i];
            }
        }
    }
}

// ---- Launch Wrapper ----
void launch_gemm_ptx(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K)
{
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    gemm_ptx_hybrid<<<grid, block>>>(A, B, C, M, N, K);
}
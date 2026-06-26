#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define BLOCK_M 64
#define BLOCK_N 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

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

    int warpRow = warpId / 4;  // 4 warps per row
    int warpCol = warpId % 4;

    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K)
    {
        // Load A tile
        int row = threadIdx.x / WMMA_K;
        int col = threadIdx.x % WMMA_K;

        if (row < BLOCK_M)
        {
            int gRow = blockRow + row;
            int gCol = k0 + col;

            As[row][col] =
                (gRow < M && gCol < K) ?
                A[gRow*K + gCol] : __float2half(0.0f);
        }

        // Load B tile
        if (row < WMMA_K)
        {
            int gRow = k0 + row;
            int gCol = blockCol + col;

            Bs[row][col] =
                (gRow < K && gCol < N) ?
                B[gRow*N + gCol] : __float2half(0.0f);
        }

        __syncthreads();

        // Compute per warp 16x16 tile
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> b_frag;

        int aRow = warpRow * WMMA_M;
        int bCol = warpCol * WMMA_N;

        wmma::load_matrix_sync(a_frag, &As[aRow][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][bCol], BLOCK_N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store result
    int cRow = blockRow + warpRow * WMMA_M;
    int cCol = blockCol + warpCol * WMMA_N;

    if (cRow < M && cCol < N)
    {
        wmma::store_matrix_sync(&C[cRow*N + cCol],
                                c_frag, N,
                                wmma::mem_row_major);
    }
}
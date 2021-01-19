#include <cstdlib>
#include <iostream>
#include "cuda_lib.h"

using namespace std;

typedef float type;
// typedef double type

const size_t WidthA = 3200;
const size_t HeightA = 3200;

const size_t WidthB = 3200;
const size_t HeightB = 3200;


/* CUDA block size */
const size_t BlockSizeX = 32;
const size_t BlockSizeY = 32;

type A[HeightA][WidthA], B[HeightB][WidthB], C[HeightA][WidthB];

__global__ void gemm(type *_C, type *_A, type *_B, size_t _wA, size_t _wB)
{
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y
    size_t y = by * blockDim.y + ty;
    size_t x = bx * blockDim.x + tx;
    size_t sizeX = gridDim.x * blockDim.x;
    size_t idx = y * sizeX + x;
    type _c = 0;
#pragma unroll
    for (size_t k = 0; k < _wA; ++k)
    {
        _c += _A[y + k] * _B[x + k * sizeX];
    }
    _C[idx] = _c;
}

__global__ void gemm_sharedmem(type *_C, type *_A, type *_B, size_t _wA, size_t _wB)
{

}

void gemm_helper()
{
    /* 设置网格grid和block */
    dim3 gridSize(WidthB / BlockSizeX, HeightA / BlockSizeY, 1);
    dim3 blockSize(BlockSizeX, BlockSizeY, 1);

    /* 分配内存 */
    type *cu_C, *cu_A, *cu_B;
    cudaMalloc((void**)&cu_C, HeightA * WidthB * sizeof(type));
    cudaMalloc((void**)&cu_A, HeightA * WidthA * sizeof(type));
    cudaMalloc((void**)&cu_B, HeightB * WidthB * sizeof(type));

    /* 传输数据 */
    cudaMemcpy(cu_A, _A, HeightA * WidthA * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_B, _B, HeightB * WidthB * sizeof(type), cudaMemcpyHostToDevice);

    /* 执行内核 */
    cout << "start kernel on GPU ..." << endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start, 0);
    gemm<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, WidthA, WidthB);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "kernel elapsed " << elapsed << " ms, ";
    std::cout << "the speed is " << (float)(HeightA * WidthB * 2 * WidthA) * 1000 / elapsed / (float)(1 << 30) << " GFlops" << std::endl;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    /* 释放内存 */
    cudaFree(cu_B);
    cudaFree(cu_A);
    cudaFree(cu_C);
}

int main()
{
    gemm_helper();
    return 0;
}
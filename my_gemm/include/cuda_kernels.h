#pragma once

#include <cuda_runtime.h>
#include "config.h"

__global__ void cuda_gemm(type *_C, type *_A, type *_B,
    size_t _wA, size_t _wB,
    size_t _maxIdx)
{
    // 分块索引
    size_t bx = blockIdx.x;
    size_t by = blockDim.y;

    // 线程索引
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // 二维索引
    size_t y = by * blockDim.y + ty;
    size_t x = bx * blockDim.x + tx;

    // 一维索引
    size_t idx = y * gridDim.x * blockDim.x + x;

    // 判断当前索引是否越界
    if (idx >= _maxIdx)
    {
        return;
    }

    // 使用寄存器存储中间计算结果
    type _c = 0;
    
    // 计算
    for (size_t k = 0; k < _wA; ++k)
    {
        _c += _A[y * _wA + k] * _B[k * _wB + x];
    }

    // 计算结果写回
    _C[idx] = _c;
}

template <size_t BLOCK_SIZE_1, size_t BLOCK_SIZE_2, size_t BLOCK_SIZE_3>
__global__ void cuda_gemm2(type *_C, type *_A, type *_B,
    size_t _wA, size_t _wB,
    size_t _maxIdx)
{
    // 分块索引
    size_t bx = blockIdx.x;
    size_t by = blockDim.y;

    // 线程索引
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // 一维索引
    size_t idx = (by * blockDim.y + ty) * gridDim.x * blockDim.x + bx * blockDim.x + tx;

    // 判断当前索引是否越界
    if (idx >= _maxIdx)
    {
        return;
    }

    // 矩阵A分块的第一个索引
    size_t aBegin = by * BLOCK_SIZE_1 * _wA;

    // 矩阵A分块的最后一个索引
    size_t aEnd = aBegin + _wA;

    // 矩阵A分块索引的更新步长
    size_t aStep = BLOCK_SIZE_3;

    // 矩阵B分块的第一个索引
    size_t bBegin = bx * BLOCK_SIZE_3;

    // 矩阵B分块索引的更新步长
    size_t bStep = BLOCK_SIZE_3 * _wB;

    // 使用寄存器存储中间计算结果
    type _c = 0;
    
    for (size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep)
    {
        // 静态分配shared memory
        __shared__ type As[BLOCK_SIZE_1][BLOCK_SIZE_3];
        __shared__ type Bs[BLOCK_SIZE_3][BLOCK_SIZE_2];

        // 将数据装载到shared memory
        As[ty][tx] = _A[a + ty * _wA + tx];
        Bs[tx][ty] = _B[b + tx * _wB + ty];

        // 等待所有线程装载完毕
        __syncthreads();

        // 分块内矩阵乘
        for (size_t k = 0; k < BLOCK_SIZE_3; ++k)
        {
            _c += As[ty][k] * Bs[tx][k];
        }

        // 等待所有线程完成计算
        __syncthreads();
    }

    // 计算结果写回
    _C[idx] = _c;
}
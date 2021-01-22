#pragma once

#include <cuda_runtime.h>
#include "config.h"

__global__ void cuda_gemm(type *_C, type *_A, type *_B,
    size_t _hC, size_t _wC, size_t _wA, size_t _wB, 
    size_t _offsetY, size_t _offsetX)
{
    // 分块索引
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    // 线程索引
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // 二维索引
    size_t y = _offsetY + by * blockDim.y + ty;
    size_t x = _offsetX + bx * blockDim.x + tx;

    // 一维索引
    size_t idx = y * _wC + x;

    // 判断当前索引是否越界
    if (y < _hC && x < _wC)
    {
        // 使用寄存器存储中间计算结果
        type _c = 0;
        size_t y_wA = y * _wA;

        // 计算
        for (size_t k = 0; k < _wA; ++k)
        {
            _c += _A[y_wA + k] * _B[k * _wB + x];
        }

        // 计算结果写回
        _C[idx] = _c;
    }
}

// 还需要参照cuda_gemm修改
template <size_t BLOCK_SIZE>
__global__ void cuda_gemm_shared_mem(type *_C, type *_A, type *_B,
    size_t _hC, size_t _wC, size_t _wA, size_t _wB, 
    size_t _offsetY, size_t _offsetX)
{
    // 分块索引
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    // 线程索引
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // 二维索引
    size_t y = _offsetY + by * blockDim.y + ty;
    size_t x = _offsetX + bx * blockDim.x + tx;

    // 一维索引
    size_t idx = y * _wC + x;

    // 判断当前索引是否越界
    if (y < _hC && x < _wC)
    {
        // 矩阵A分块的第一个索引
        size_t aBegin = by * BLOCK_SIZE * _wA;

        // 矩阵A分块的最后一个索引
        size_t aEnd = aBegin + _wA;

        // 矩阵A分块索引的更新步长
        size_t aStep = BLOCK_SIZE;

        // 矩阵B分块的第一个索引
        size_t bBegin = bx * BLOCK_SIZE;

        // 矩阵B分块索引的更新步长
        size_t bStep = BLOCK_SIZE * _wB;

        // 使用寄存器存储中间计算结果
        type _c = 0;
        
        for (size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep)
        {
            // 静态分配shared memory
            __shared__ type As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ type Bs[BLOCK_SIZE][BLOCK_SIZE];

            // 将数据装载到shared memory
            As[ty][tx] = _A[a + ty * _wA + tx];
            Bs[ty][tx] = _B[b + ty * _wB + tx];

            // 等待所有线程装载完毕
            __syncthreads();

            // 分块内矩阵乘
            for (size_t k = 0; k < BLOCK_SIZE; ++k)
            {
                _c += As[ty][k] * Bs[k][tx];
            }

            // 等待所有线程完成计算
            __syncthreads();
        }

        // 计算结果写回
        _C[idx] = _c;
    }
}
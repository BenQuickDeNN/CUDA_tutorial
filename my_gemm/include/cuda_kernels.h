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

    // 使用寄存器存储中间计算结果
    type _c = 0;
    size_t y_wA = y * _wA;

    // shared memory的分块个数
    size_t blocks = _wA / BLOCK_SIZE;

    // 计算对齐部分
    for (size_t block = 0; block < blocks; ++block)
    {
        // 静态分配shared memory
        __shared__ type As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ type Bs[BLOCK_SIZE][BLOCK_SIZE];

        // 将数据装载到shared memory
        As[ty][tx] = _A[y_wA + block * BLOCK_SIZE + tx];
        Bs[ty][tx] = _B[(block * BLOCK_SIZE + tx) * _wB + x];

        // 等待所有线程装载完毕
        __syncthreads();

        // 判断当前索引是否越界
        if (y < _hC && x < _wC)
        {
            // 分块内矩阵乘
            for (size_t k = 0; k < BLOCK_SIZE; ++k)
            {
                _c += As[ty][k] * Bs[k][tx];
            }
         }

        // 等待所有线程完成计算
        __syncthreads();
    }

    // 判断当前索引是否越界
    if (y < _hC && x < _wC)
    {
        // 计算非对齐部分
        for (size_t k = blocks * BLOCK_SIZE; k < _wA; ++k)
        {
            _c += _A[y_wA + k] * _B[k * _wB + x];
        }
        // 计算结果写回
        _C[idx] = _c;
    }
}
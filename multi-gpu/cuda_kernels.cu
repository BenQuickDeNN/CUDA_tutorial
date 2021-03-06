#pragma once

#include "cuda_lib.cu"
#include "configure.hpp"

/**
 * @brief CUDA版通用矩阵乘
 * @param _C 储存结果的矩阵
 * @param _A 参与运算的矩阵
 * @param _B 参与运算的矩阵
 * @param _height 矩阵C的高度
 * @param _width 矩阵C的宽度
 * @param _width_A 矩阵A的宽度和矩阵B的高度
 */
__global__ void cuda_my_gemm(type *_C, type *_A, type *_B, size_t _height, size_t _width, size_t _width_A)
{

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 线程号
    size_t stride = gridDim.x * blockDim.x; // 索引号更新的步长
    size_t max_idx = _height * _width; // 最大索引号
    while (idx < max_idx)
    {
        size_t h = idx / _width;
        size_t w = idx % _width;
        size_t idx1= h * _width + w;
        size_t idx2 = h * _width_A;
        // _C[idx1] = 0.0;
        for (size_t k = 0; k < _width_A; ++k)
        {
            _C[idx1] += _A[idx2 + k] * _B[k * _width + w];
        }
        idx += stride;
    }
}
#pragma once

#include "cuda_lib.cu"
#include "configure.hpp"

/**
 * @brief 启动函数
 * @param _C 储存结果的矩阵
 * @param _A 参与运算的矩阵
 * @param _B 参与运算的矩阵
 * @param _height 矩阵C的高度
 * @param _width 矩阵C的宽度
 * @param _offset_h 矩阵C竖直方向的起始索引
 * @param _width_A 矩阵A的宽度和矩阵B的高度
 * @param _width_B 矩阵B的宽度
 * @param _device_id GPU编号号
 */
void cuda_exec_gemm(type *_C, type *_A, type *_B, 
    size_t _height, size_t _width, 
    size_t _offset_h, size_t _width_A, size_t _width_B,
    size_t _device_id)
{
    cudaSetDevice(_device_id); // 设置用于计算的GPU

    /* 设置网格grid和block */
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int num_blocks = devProp.multiProcessorCount * devProp.maxBlocksPerMultiProcessor;
    int num_threads_per_blocks = devProp.maxThreadsPerBlock;
    dim3 gridSize(num_blocks, 1, 1);
    dim3 blockSize(num_threads_per_blocks, 1, 1);

    /* 分配内存 */
    type *cu_C, *cu_A, *cu_B;
    cudaMalloc((void**)&cu_C, _height * _width * sizeof(type));
    cudaMalloc((void**)&cu_A, _height * _width_A * sizeof(type));
    cudaMalloc((void**)&cu_B, _width * _width_B * sizeof(type));

    /* 释放内存 */
    cudaFree(cu_A);
    cudaFree(cu_C);
}
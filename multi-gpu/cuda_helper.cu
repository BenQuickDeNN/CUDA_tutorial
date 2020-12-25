#pragma once

#include <iostream>

#include "cuda_lib.cu"
#include "configure.hpp"
#include "cuda_kernels.cu"

/**
 * @brief 启动函数
 * @param _C 储存结果的矩阵
 * @param _A 参与运算的矩阵
 * @param _B 参与运算的矩阵
 * @param _height 矩阵C的高度
 * @param _width 矩阵C的宽度
 * @param _offset_h 矩阵C竖直方向的起始索引
 * @param _width_A 矩阵A的宽度和矩阵B的高度
 * @param _device_id GPU编号号
 */
void cuda_exec_gemm(type *_C, type *_A, type *_B, 
    size_t _height, size_t _width, 
    size_t _offset_h, size_t _width_A,
    size_t _device_id)
{
    if (cudaSetDevice(_device_id) == cudaError::cudaErrorNoDevice)
    {
        std::cerr << "cuda error: no device!" << std::endl;
        return;
    }

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
    cudaMalloc((void**)&cu_B, _width_A * _width * sizeof(type));

    /* 传输数据 */
    cudaMemcpy(cu_A, _A, _height * _width_A * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_B, _B, _width_A * _width * sizeof(type), cudaMemcpyHostToDevice);

    /* 执行kernel */
    std::cout << "running kernel on GPU " << _device_id << " ..." << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start, 0);
    cuda_my_gemm<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, _height, _width, _offset_h, _width_A, _width_B);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "kernel on GPU " << _device_id <<" elapsed " << elapsed << " ms" << std::endl;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    /* 传回结果 */
    cudaMemcpy(_C, cu_C, _height * _width * sizeof(type), cudaMemcpyDeviceToHost);

    /* 释放内存 */
    cudaFree(cu_B);
    cudaFree(cu_A);
    cudaFree(cu_C);
}
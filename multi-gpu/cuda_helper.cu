#pragma once

#include <iostream>

#include "cuda_lib.cu"
#include "configure.hpp"
#include "cuda_kernels.cu"

const size_t GIGA = 1024 * 1024 * 1024;

/**
 * @brief 启动函数
 * @param _C 储存结果的矩阵
 * @param _A 参与运算的矩阵
 * @param _B 参与运算的矩阵
 * @param _height 矩阵C的高度
 * @param _width 矩阵C的宽度
 * @param _width_A 矩阵A的宽度和矩阵B的高度
 * @param _device_id GPU编号
 */
bool cuda_exec_gemm(type *_C, type *_A, type *_B, 
    size_t _height, size_t _width, size_t _width_A,
    size_t _device_id)
{
    if (cudaSetDevice(_device_id) != 0)
    {
        omp_set_lock(&omplock);
        std::cerr << "error: GPU " << _device_id << " is not available" << std::endl;
        omp_unset_lock(&omplock);
        return false;
    }

    /* 设置网格grid和block */
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, _device_id);
    size_t total_workload = (_height * _width + _height * _width_A + _width_A * _width) * sizeof(type);

    omp_set_lock(&omplock);
    std::cout << "total workload on GPU " << _device_id << " is " << 
        (double)total_workload / (double)GIGA << " GB" << std::endl;
    omp_unset_lock(&omplock);
    if (total_workload > devProp.totalGlobalMem) // 检查工作负载是否超过内存
    {
        omp_set_lock(&omplock);
        std::cerr << "error: too large total workload on GPU " << _device_id << 
            ". Only " << (double)devProp.totalGlobalMem  / (double)GIGA <<
            "GB is available on GPU " << _device_id << std::endl;
        omp_unset_lock(&omplock);
        return false;
    }

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
    omp_set_lock(&omplock);
    std::cout << "running kernel on GPU " << _device_id << " ..." << std::endl;
    omp_unset_lock(&omplock);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start, 0);
    cuda_my_gemm<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, _height, _width, _width_A);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "kernel on GPU " << _device_id <<" elapsed " << elapsed << " ms, ";
    std::cout << "the speed is " << (float)(_height * _width * 2 * _width_A) * 1000 / elapsed / (float)(1 << 30) << " GFlops" << std::endl;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    /* 传回结果 */
    cudaMemcpy(_C, cu_C, _height * _width * sizeof(type), cudaMemcpyDeviceToHost);

    /* 释放内存 */
    cudaFree(cu_B);
    cudaFree(cu_A);
    cudaFree(cu_C);

    return true;
}
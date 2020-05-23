/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CUDA_JACOBI_1D_CU

#define CUDA_JACOBI_1D_CU

#include "configure.h"
#include "cuda_jacobi-1d_kernel.cu"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

/**
 * @brief 内核驱动函数
 */
void cuda_jacobi_1d(type* data, const type& scalar, const int& len, const int& step);

void cuda_jacobi_1d(type* data, const type& scalar, const int& len, const int& step)
{
    using namespace std;
    
    /* 设置线程布局 */
    dim3 gridSize(NUM_SM, 1, 1);
    dim3 blockSize(MAX_NUM_THREAD_PER_SM, 1, 1);

    /* GPU可提供的总线程数 */
    const int total_num_threads = NUM_SM * MAX_NUM_THREAD_PER_SM;

    /* 每个线程处理的数据规模 */
    const int batSize = std::ceil((float)len / (float)total_num_threads);

    /* 分配给GPU的内存大小 */
    const int cuda_mem = batSize * total_num_threads;

    /* 分配GPU内存 */
    type* cu_data;
    fprintf(stderr, "cudaMalloc error code = %d\n", cudaMalloc((void**)&cu_data, 2 * cuda_mem * sizeof(type)));

    /* 传输数据 从host到device */
    fprintf(stderr, "cudaMemcpy error code = %d\n", cudaMemcpy(cu_data, data, 2 * len * sizeof(type), cudaMemcpyHostToDevice));

    /* 启动内核 */
    cuda_jacobi_1d_kernel<<<gridSize, blockSize>>>(cu_data, scalar, cuda_mem, step, batSize);

    /* 传输数据 从device到host */
    fprintf(stderr, "cudaMemcpy error code = %d\n", cudaMemcpy(data, cu_data, 2 * len * sizeof(type), cudaMemcpyDeviceToHost));

    /* 释放GPU内存 */
    fprintf(stderr, "cudaFree error code = %d\n", cudaFree(cu_data));
}

#endif
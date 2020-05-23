/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CUDA_JACOBI_1D_KERNEL_CU

#define CUDA_JACOBI_1D_KERNEL_CU

#include "configure.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

/**
 * @brief 1维jacobi迭代CUDA内核
 */ 
__global__ void cuda_jacobi_1d_kernel(type* data, type scalar, int len, int step, int batSize);

__global__ void cuda_jacobi_1d_kernel(type* data, type scalar, int len, int step, int batSize)
{
    const int iStart = (threadIdx.x + blockIdx.x * blockDim.x) * batSize;
    const int iEnd = iStart + batSize;

    for (int t = 0; t < step; t++)
    {
        const int tIdx = (t % 2) * len;
        for (int i = iStart; i < iEnd; i++)
            data[(t % 2 + 1) * len + i] += scalar * data[tIdx + i] + 
                data[tIdx + (i - 1) % len] + data[tIdx + (i + 1) % len];
        // 每个时间步都要对所有线程进行同步
        //cudaThreadSynchronize();
        //__syncthreads();
        __threadfence();
    }
}

#endif
/*********************************************************************
 * @file 	cuda_vadd_kernel.cu
 * @brief 	kernel source of vector add
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * you can reedit or modify this file
*********************************************************************/

#ifndef CUDA_VADD_KERNEL_H

#define CUDA_VADD_KERNEL_H

#include "configure.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
* @brief cuda kernel -- vector add: c = alpha * a + beta * b
* @param c output vector c
* @param a input vector a
* @param b input vector b
* @param alpha scalar alpha
* @param beta scalar beta
* @param batSize batch size
*/
__global__ void kernel_vadd(type* c, const type* a, const type* b, 
    type alpha, type beta, int batSize);

__global__ void kernel_vadd(type* c, const type* a, const type* b, 
    type alpha, type beta, int batSize)
{
    /* compute thread id */
    int iStart = (threadIdx.x + blockIdx.x * blockDim.x) * batSize;
    const int iEnd = iStart + batSize;

    /* compute kernel */
    for (int i = iStart; i < iEnd; i++)
        c[i] += alpha * a[i] + beta * b[i];
}

#endif
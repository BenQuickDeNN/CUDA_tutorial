/*********************************************************************
 * @file 	vadd_kernel.cu
 * @brief 	kernel source of vector add
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * you can reedit or modify this file
*********************************************************************/

#ifndef VADD_KERNEL_H

#define VADD_KERNEL_H

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
template<class T>
__global__ void kernel_vadd(T* c, const T* a, const T* b, 
    T alpha, T beta, int batSize);

template<class T>
__global__ void kernel_vadd(T* c, const T* a, const T* b, 
    T alpha, T beta, int batSize)
{
    /* compute thread id */
    int iStart = (threadIdx.x + blockIdx.x * blockDim.x) * batSize;
    const int iEnd = iStart + batSize;

    /* compute kernel */
    for (int i = iStart; i < iEnd; i++)
        c[i] += alpha * a[i] + beta * b[i];
}

#endif
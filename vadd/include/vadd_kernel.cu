/*********************************************************************
 * @file 	vadd_kernel.cu
 * @brief 	kernel source of vector add
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2020-3-5
 * you can reedit or modify this file
*********************************************************************/

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
    T alpha, T beta, int batSize, int steps);

template<class T>
__global__ void kernel_vadd(T* c, const T* a, const T* b, 
    T alpha, T beta, int batSize, int steps)
{
    /* compute thread id */
    int iStart = threadIdx.x + blockIdx.x * blockDim.x;
    iStart *= batSize;
    const int iEnd = iStart + batSize;

    /* compute kernel */
    for (int s = 0; s < steps; s++)
        for (int i = iStart; i < iEnd; i++)
            c[i] += alpha * a[i] + beta * b[i];
}
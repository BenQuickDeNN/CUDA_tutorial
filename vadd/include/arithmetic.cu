/*********************************************************************
 * @file 	arithmetic.cu
 * @brief 	arithmetic file, including vadd and mmul
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2020-3-5
 * you can reedit or modify this file
*********************************************************************/

#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
* @brief main function to activate vadd kernel
* @param c vector c
* @param a vector a
* @param b vector b
* @param alpha scalar alpha
* @param beta scalar beta
* @param batSize batch size
*/
template<class T>
void cuda_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& batSize, const int& steps, 
    const dim3& gridSize, const dim3& blockSize, const int& len);

template<class T>
void cuda_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& batSize, const int& steps, 
    const dim3& gridSize, const dim3& blockSize, const int& len)
{
    if (c == nullptr || a == nullptr || b == nullptr)
    {
        std::fprintf(stderr, "vadd error: vector empty!\r\n");
        return;
    }

    /* allocate memory on gpu */
    T *cu_c, *cu_a, *cu_b;

    cudaMalloc((void**)&cu_c, len * sizeof(T));
    cudaMalloc((void**)&cu_a, len * sizeof(T));
    cudaMalloc((void**)&cu_b, len * sizeof(T));

    /* copy data from host to gpu */
    cudaMemcpy(cu_a, a, len * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_b, b, len * sizeof(T), cudaMemcpyHostToDevice);

    /* activate kernel */
    kernel_vadd<<<gridSize, blockSize>>>(cu_c, cu_a, cu_b, alpha, beta, batSize, steps);

    /* copy data from gpu to host */
    cudaMemcpy(c, cu_c, len * sizeof(T), cudaMemcpyDeviceToHost);
    
    /* free memory on gpu */
    cudaFree(cu_b);
    cudaFree(cu_a);
    cudaFree(cu_c);
}
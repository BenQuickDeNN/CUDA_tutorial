/*********************************************************************
 * @file 	arithmetic.cu
 * @brief 	arithmetic file, including vadd and mmul
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * you can reedit or modify this file
*********************************************************************/

#ifndef CUDA_VADD_H

#define CUDA_VADD_H

#include "cuda_vadd_kernel.h"
#include "timer_win.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int NUM_SM = 16; // SM的个数
const int MAX_NUM_THREAD_PER_SM = 1024; // 每个SM中允许的最大线程数

/**
* @brief main function to activate vadd kernel
* @param c vector c
* @param a vector a
* @param b vector b
* @param alpha scalar alpha
* @param beta scalar beta
* @param batSize batch size
* @param elapsed 计算时间
*/
template<class T>
void cuda_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& len, double& elapsed);

template<class T>
void cuda_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& len, double& elapsed)
{
    using namespace std;

    if (c == nullptr || a == nullptr || b == nullptr)
    {
        std::fprintf(stderr, "vadd error: vector empty!\r\n");
        return;
    }

    /* 定义grid和block的尺寸 */
    dim3 gridSize(NUM_SM, 1, 1);
    dim3 blockSize(MAX_NUM_THREAD_PER_SM, 1, 1);

    /* 计算batSize，即每个线程处理的范围 */
    int batSize, lenCuda;
    if (len % (NUM_SM * MAX_NUM_THREAD_PER_SM) != 0) // 不对齐
    {
        batSize = len / (NUM_SM * MAX_NUM_THREAD_PER_SM) + 1;
        lenCuda = batSize * NUM_SM * MAX_NUM_THREAD_PER_SM;
    }
    else // 对齐
    {
        batSize = len / (NUM_SM * MAX_NUM_THREAD_PER_SM);
        lenCuda = len;
    }

    /* allocate memory on gpu */
    T *cu_c, *cu_a, *cu_b;

    cudaMalloc((void**)&cu_c, lenCuda * sizeof(T));
    cudaMalloc((void**)&cu_a, lenCuda * sizeof(T));
    cudaMalloc((void**)&cu_b, lenCuda * sizeof(T));

    /* copy data from host to gpu */
    cudaMemcpy(cu_a, a, len * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_b, b, len * sizeof(T), cudaMemcpyHostToDevice);

    /* activate kernel */
    Timer_win tw;
    tw.start();
    kernel_vadd<<<gridSize, blockSize>>>(cu_c, cu_a, cu_b, alpha, beta, batSize);
    elapsed = tw.endus();

    /* copy data from gpu to host */
    cudaMemcpy(c, cu_c, len * sizeof(T), cudaMemcpyDeviceToHost);
    
    /* free memory on gpu */
    cudaFree(cu_b);
    cudaFree(cu_a);
    cudaFree(cu_c);
    
}

#endif
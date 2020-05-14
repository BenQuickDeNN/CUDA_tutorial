/*********************************************************************
 * @file 	arithmetic.cu
 * @brief 	arithmetic file, including vadd and mmul
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * you can reedit or modify this file
*********************************************************************/

#ifndef CUDA_VADD_H

#define CUDA_VADD_H

#include "vadd_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>

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
*/
template<class T>
void cuda_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& len);

template<class T>
void cuda_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& len)
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
    LARGE_INTEGER start, end, frequency; // 用于记录时间
    QueryPerformanceFrequency(&frequency); // 获取硬件时钟频率
    QueryPerformanceCounter(&start); // 开始
    kernel_vadd<<<gridSize, blockSize>>>(cu_c, cu_a, cu_b, alpha, beta, batSize);
    QueryPerformanceCounter(&end); // 结束

    /* copy data from gpu to host */
    cudaMemcpy(c, cu_c, len * sizeof(T), cudaMemcpyDeviceToHost);
    
    /* free memory on gpu */
    cudaFree(cu_b);
    cudaFree(cu_a);
    cudaFree(cu_c);
    
    /* 计算平均速度 */
    double elapsed = (double)(end.QuadPart - start.QuadPart) / ((double)frequency.QuadPart);
    printf("CUDA arrayadd elapsed = %.1f us\r\n", elapsed * 1000000.0);
    double speed = (((double)len) / elapsed) / 1000000000.0;
    if (sizeof(T) == sizeof(float))
        fprintf(stdout, "the speed of CUDA arrayadd is %.3f GFLOPS \r\n", speed);
    else if (sizeof(T) == sizeof(double))
        fprintf(stdout, "the speed of CUDA arrayadd is %.3f GDFLOPS \r\n", speed);
}

#endif
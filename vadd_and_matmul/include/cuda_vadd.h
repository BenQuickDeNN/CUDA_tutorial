/*********************************************************************
 * @file 	cuda_vadd.h
 * @brief 	cuda driven file
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * you can reedit or modify this file
*********************************************************************/

#ifndef CUDA_VADD_H

#define CUDA_VADD_H

#include "cuda_vadd_kernel.h"
#include "configure.h"
#include "timer_win.h"

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
* @param len host端数组长度
* @param elapsed 计算时间
*/
void cuda_vadd(type* c, const type* a, const type* b,
    const type& alpha, const type& beta, const int& len, double& elapsed);

void cuda_vadd(type* c, const type* a, const type* b,
    const type& alpha, const type& beta, const int& len, double& elapsed)
{
    using namespace std;

    if (c == nullptr || a == nullptr || b == nullptr)
    {
        fprintf(stderr, "vadd error: vector empty!\r\n");
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
    type *cu_c, *cu_a, *cu_b;
    cudaMalloc((void**)&cu_c, lenCuda * sizeof(type));
    cudaMalloc((void**)&cu_a, lenCuda * sizeof(type));
    cudaMalloc((void**)&cu_b, lenCuda * sizeof(type));

    /* copy data from host to gpu */
    cudaMemcpy(cu_a, a, len * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_b, b, len * sizeof(type), cudaMemcpyHostToDevice);

    /* activate kernel */
    Timer_win tw;
    tw.start();
    kernel_vadd<<<gridSize, blockSize>>>(cu_c, cu_a, cu_b, alpha, beta, batSize);
    elapsed = tw.endus();

    /* copy data from gpu to host */
    cudaMemcpy(c, cu_c, len * sizeof(type), cudaMemcpyDeviceToHost);
    
    /* free memory on gpu */
    cudaFree(cu_b);
    cudaFree(cu_a);
    cudaFree(cu_c);
}

#endif
/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "configure.h"

/**
 * @brief CUDA计算矩阵乘内核，按矩阵行划分
 * @param widthC 矩阵C的宽
 * @param widthA 矩阵A的宽
 * @param widthB 矩阵B的宽
 */
__global__ void cuda_matmul_kernel(type* C, const type* A, const type* B, 
    int widthC, int widthA, int widthB, int batSizeC);

__global__ void cuda_matmul_kernel(type* C, const type* A, const type* B, 
    int widthC, int widthA, int widthB, int batSizeC)
{
    /* 计算线程id */
    const int iStart = (threadIdx.x + blockIdx.x * blockDim.x) * batSizeC;
    const int iEnd = iStart + batSizeC;
    int h, w;

    for (int i = iStart; i < iEnd; i++)
    {
        h = i / widthC;
        w = i % widthC;
        for (int k = 0; k < widthA; k++)
            C[h * widthC + w] += A[h * widthA + k] * B[k * widthB + w];
    }
}

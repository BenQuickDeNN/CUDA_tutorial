/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CUDA_MATMUL_ROWS

#define CUDA_MATMUL_ROWS

#include "matrix.h"
#include "cuda_matmul_kernel.h"
#include "configure.h"

#include <cmath>

/**
 * @brief CUDA矩阵乘
 */
void cuda_matmul_rows(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& h_start, const int& h_end);

void cuda_matmul_rows(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& h_start, const int& h_end)
{
    dim3 gridSize(NUM_SM, 1, 1);
    dim3 blockSize(MAX_NUM_THREAD_PER_SM, 1, 1);

    /* 计算GPU负责的行数 */
    const int rows = h_end - h_start;
    printf("rows=%d\n", rows);

    /* 计算每个线程负责矩阵的元素个数 */
    const int batSizeC = (int)std::ceil((double)(rows * C.getWidth()) / 
        (double)(NUM_SM * MAX_NUM_THREAD_PER_SM));
    const int batSizeA = (int)std::ceil((double)(rows * A.getWidth()) / 
        (double)(NUM_SM * MAX_NUM_THREAD_PER_SM));
    const int batSizeB = (int)std::ceil((double)(B.getHeight() * B.getWidth()) / 
        (double)(NUM_SM * MAX_NUM_THREAD_PER_SM));

    printf("batSizeC=%d, batSizeA=%d, batSizeB=%d\n", batSizeC, batSizeA, batSizeB);

    /* 计算需要在GPU上分配的内存大小 */
    const int len_cudaC = batSizeC * NUM_SM * MAX_NUM_THREAD_PER_SM;
    const int len_cudaA = batSizeA * NUM_SM * MAX_NUM_THREAD_PER_SM;
    const int len_cudaB = batSizeB * NUM_SM * MAX_NUM_THREAD_PER_SM;

    printf("lencudaC=%d, len_cudaA=%d, len_cudaB=%d\n", len_cudaC, len_cudaA, len_cudaB);

    /* 分配内存 */
    type *cuC, *cuA, *cuB;
    printf("cuC malloc state = %d\n", cudaMalloc((void**)&cuC, len_cudaC * sizeof(type)));
    printf("cuA malloc state = %d\n", cudaMalloc((void**)&cuA, len_cudaA * sizeof(type)));
    printf("cuB malloc state = %d\n", cudaMalloc((void**)&cuB, len_cudaB * sizeof(type)));

    /* 传输数据 */
    printf("cuA cpy state = %d\n", cudaMemcpy(cuA, &A(h_start, 0), rows * A.getWidth() * sizeof(type), cudaMemcpyHostToDevice));
    printf("cuB cpy state = %d\n", cudaMemcpy(cuB, &B(0, 0), B.getHeight() * B.getWidth() * sizeof(type), cudaMemcpyHostToDevice));

    /* 启动内核 */
    cuda_matmul_kernel<<<gridSize, blockSize>>>(cuC, cuA, cuB, 
        C.getWidth(), A.getWidth(), B.getWidth(), batSizeC);

    /* 传回结果 */
    printf("cuC cpy state = %d\n", cudaMemcpy(&C(h_start, 0), cuC, rows * C.getWidth() * sizeof(type), cudaMemcpyDeviceToHost));

    /* 释放内存 */
    cudaFree(cuB);
    cudaFree(cuA);
    cudaFree(cuC);

    C.disp(h_start, h_start + 1);
}

#endif
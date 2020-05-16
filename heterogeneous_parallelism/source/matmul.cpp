/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#include "configure.h"
#include "cpu_matmul.h"
#include "cuda_matmul.h"

#include <omp.h>
#include <cstdio>

const int HEIGHT = 1 << 4;
const int WIDTH = 1 << 4;

int main()
{
    using namespace std;

    Matrix C1(HEIGHT, WIDTH), C2(HEIGHT, WIDTH), A(HEIGHT, WIDTH), B(HEIGHT, WIDTH);
    A.randFill(10);
    B.randFill(10);

    printf("serial:\n");
    C1.fill(0.0);
    mat_mul(C1, A, B);
    C1.disp();

    printf("openmp + cuda:\n");
    C2.fill(0.0);
    const int batSize = 3; // 每个CPU OpenMP线程处理的行数
    #pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < NUM_THREAD; i++)
    {
        const int h_start = i * batSize;
        const int h_end = h_start + batSize;
        if (i == NUM_HOST_THREAD) // 让最后一个线程去调用GPU，计算矩阵C剩下的行
            cuda_matmul_rows(C2, A, B, h_start, HEIGHT);
        else
            mat_mul_rows(C2, A, B, h_start, h_end);
    }
    C2.disp();

    if (C1.isEqual(C2, 1.0))
        printf("C1 and C2 are equal.\n");
    else
        printf("C1 and C2 are not equal!\n");
    
    return 0;
}
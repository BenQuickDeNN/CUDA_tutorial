/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#include "configure.h"
#include "cpu_matmul.h"
#include "cuda_matmul.h"
#include "timer_win.h"

#include <omp.h>
#include <cstdio>
#include <algorithm>

const int HEIGHT = 1 << 10;
const int WIDTH = 1 << 10;

const int OMP_HEIGHT = HEIGHT / 4; // CPU负责的矩阵行数

int main()
{
    using namespace std;

    Matrix C1(HEIGHT, WIDTH), C2(HEIGHT, WIDTH), A(HEIGHT, WIDTH), B(WIDTH, WIDTH);
    A.randFill(2);
    B.randFill(2);
    Timer_win tw;

    /* OpenMP */
    printf("openmp:\n");
    int batSize = (int)ceil((double)HEIGHT / (double)NUM_THREAD);
    C1.fill(0.0);
    tw.start();
    #pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < NUM_THREAD; i++)
    {
        const int h_start = i * batSize;
        const int h_end = h_start + batSize;
        mat_mul_rows(C1, A, B, h_start, min(h_end, HEIGHT));
    }
    printf("elapsed %f s\n", tw.ends());

    /* OpenMP + CUDA */
    printf("openmp + cuda:\n");
    batSize = OMP_HEIGHT / NUM_HOST_THREAD; // 每个CPU OpenMP线程处理的行数
    C2.fill(0.0);
    tw.start();
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
    printf("elapsed %f s\n", tw.ends());

    /* 验证结果正确性 */
    if (C1.isEqual(C2, 1.0))
        printf("C1 and C2 are equal.\n");
    else
        printf("C1 and C2 are not equal!\n");
    
    return 0;
}
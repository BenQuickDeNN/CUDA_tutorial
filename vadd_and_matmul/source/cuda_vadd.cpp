/*********************************************************************
 * @file 	vadd.cu
 * @brief 	main source file
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2020-3-5
 * you can reedit or modify this file
*********************************************************************/

#include "cuda_vadd.h"
#include "timer_win.h"
#include "configure.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

type a[LEN], b[LEN], c[LEN];

int main(int argc, char** argv)
{
    double elapsed_cuda, elapsed_cuda_trans;
    /* 执行CUDA 数组加 */
    Timer_win tw;
    tw.start();
    cuda_vadd<type>(c, a, b, alpha, beta, LEN, elapsed_cuda);
    elapsed_cuda_trans = tw.endus();

    /* 打印运行时间 */
    printf("CUDA arrayadd elapsed = %.3f us\r\n", elapsed_cuda);
    printf("CUDA arrayadd elapsed (including data transmission) = %.3f us\r\n", 
        elapsed_cuda_trans);

    /* 浮点计算速率 */
    double speed_cuda = (double)LEN / elapsed_cuda / THOUSAND;
    double speed_cuda_trans = (double)LEN / elapsed_cuda_trans / THOUSAND;
    if (sizeof(type) == sizeof(float))
    {
        fprintf(stdout, "the speed of CUDA arrayadd is %.3f GFLOPS \r\n", speed_cuda);
        fprintf(stdout, "the speed of CUDA arrayadd (including data transmission) is %.3f GFLOPS \r\n", 
            speed_cuda_trans);
    }
    else if (sizeof(type) == sizeof(double))
    {
        fprintf(stdout, "the speed of CUDA arrayadd is %.3f GDFLOPS \r\n", speed_cuda);
        fprintf(stdout, "the speed of CUDA arrayadd (including data transmission) is %.3f GDFLOPS \r\n", 
            speed_cuda_trans);
    }

    return 0;
}
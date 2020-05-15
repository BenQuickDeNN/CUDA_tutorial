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
const type alpha = 2.0;
const type beta = 3.0;

int main(int argc, char** argv)
{
    using namespace std;

    double elapsed;
    /* 执行CUDA 数组加 */
    Timer_win tw;
    tw.start();
    cuda_vadd(c, a, b, alpha, beta, LEN, elapsed);
    double elapsed_trans = tw.endus();

    /* 打印运行时间 */
    printf("CUDA arrayadd elapsed = %.3f us\r\n", elapsed);
    printf("CUDA arrayadd elapsed (including data transmission) = %.3f us\r\n", 
        elapsed_trans);

    /* 浮点计算速率 */
    double speed = (double)LEN / elapsed;
    double speed_trans = (double)LEN / elapsed_trans;
    if (sizeof(type) == sizeof(float))
    {
        fprintf(stdout, "the speed of CUDA arrayadd is %.3f MFLOPS \r\n", speed);
        fprintf(stdout, "the speed of CUDA arrayadd (including data transmission) is %.3f MFLOPS \r\n", 
            speed_trans);
    }
    else if (sizeof(type) == sizeof(double))
    {
        fprintf(stdout, "the speed of CUDA arrayadd is %.3f MDFLOPS \r\n", speed);
        fprintf(stdout, "the speed of CUDA arrayadd (including data transmission) is %.3f MDFLOPS \r\n", 
            speed_trans);
    }

    return 0;
}
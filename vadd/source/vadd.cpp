/*********************************************************************
 * @file 	vadd.cu
 * @brief 	main source file
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2020-3-5
 * you can reedit or modify this file
*********************************************************************/

#include "cuda_vadd.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef float type;

const int LEN = 1 << 20; // 2^20

const type alpha = 2.0, beta = 3.0;
type a[LEN], b[LEN], c1[LEN], c2[LEN];

int main(int argc, char** argv)
{
    /* 执行CPU 数组加 */

    /* 执行CUDA 数组加。其数据传输时间不计入运行时间 */
    cuda_vadd<type>(c2, a, b, alpha, beta, LEN);

    /* 验证结果 */

    return 0;
}
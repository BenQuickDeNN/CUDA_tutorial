/*********************************************************************
 * @file 	vadd.cu
 * @brief 	main source file
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2020-3-5
 * you can reedit or modify this file
*********************************************************************/

#include "arithmetic.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef float type;

type *c, *a, *b, *c_check;

int main(int argc, char** argv)
{
    /* 检查输入的参数个数 */
    if (argc != 5)
    {
        std::fprintf(stderr, "error! you have input %d parameter(s). you must input 5 parameters!\r\n", argc);
        return -1;
    }

    /* 定义标量 */
    const int LEN = std::atoi(argv[1]);
    const int STEPS = std::atoi(argv[2]);
    const int BAT_SIZE = std::atoi(argv[3]);
    const int BLOCK_X = std::atoi(argv[4]);
    const type alpha = 1.0;
    const type beta = 1.0;
    const type error = 0.1;
    clock_t t_start, t_end;
    float elapsed;

    /* 分配内存 */
    c = (type*)std::malloc(LEN * sizeof(type));
    a = (type*)std::malloc(LEN * sizeof(type));
    b = (type*)std::malloc(LEN * sizeof(type));
    c_check = (type*)std::malloc(LEN * sizeof(type));

    /* 使用随机法初始化数据 */
    for (int i = 0; i < LEN; i++)
    {
        a[i] = std::rand() / (2 * STEPS);
        b[i] = std::rand() / (2 * STEPS);
        c[i] = 0.0;
        c_check[i] = 0.0;
    }

    /* 计算标准答案，并测量baseline运行时间 */
    std::fprintf(stdout, "start computing baseline...\r\n");
    t_start = clock();
    for (int s = 0; s < STEPS; s++)
        for (int i = 0; i < LEN; i++)
            c_check[i] += alpha * a[i] + beta * b[i];
    t_end = clock();
    elapsed = (float)(t_end - t_start) / CLK_TCK;
    std::fprintf(stdout, "baseline elapsed %f s\r\n", elapsed);

    /* 计算所需Grid的尺寸 */
    const int GRID_X = std::ceil(LEN / (BLOCK_X * BAT_SIZE));

    /* 设置grid和block */
    dim3 gridSize(GRID_X, 1, 1);
    dim3 blockSize(BLOCK_X, 1, 1);

    /* 启用GPU计算 */
    std::fprintf(stdout, "start computing on GPU...\r\n");
    t_start = clock();
    cuda_vadd(c, a, b, alpha, beta, BAT_SIZE, STEPS, gridSize, blockSize, LEN);
    t_end = clock();
    elapsed = (float)(t_end - t_start) / CLK_TCK;

    /* 验证结果 */
    bool isCorrect = true;
    std::fprintf(stdout, "verifying...\r\n");
    int tmpI;
    for (int i = 0; i < LEN; i++)
        if (std::abs(c[i] - c_check[i]) > error)
        {
            isCorrect = false;
            tmpI = i;
            break;
        }
    if (isCorrect)
        std::fprintf(stdout, "computing on GPU elapsed %f s\r\n", elapsed);
    else
        std::fprintf(stderr, "error! incorrected result on %d!\r\n", tmpI);

    /* 释放内存 */
    std::free(c_check);
    std::free(b);
    std::free(a);
    std::free(c);

    return 0;
}
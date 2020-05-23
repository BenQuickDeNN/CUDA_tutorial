/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#include "configure.h"
#include "cuda_jacobi-1d.cu"

#include <cstdlib>
#include <cstdio>

const int LEN = 8;
const int STEP = 10;
const type scalar = 0.5;

type data_cpu[2][LEN];
type data_gpu[2][LEN];

int main()
{
    using namespace std;

    /* 用随机数初始化 */
    for (int i = 0; i < LEN; i++)
    {
        data_cpu[0][i] = (type)((float)rand() / (float)(STEP * LEN));
        data_gpu[0][i] = data_cpu[0][i];
    }

    /* 打印结果 */
    printf("start:\n");
    for (int i = 0; i < LEN - 1; i++)
        printf("%.1f\t", data_cpu[0][i]);
    printf("%.1f\n", data_cpu[0][LEN - 1]);

    /* 使用CPU做计算 */
    for (int t = 0; t < STEP; t++)
        for (int i = 0; i < LEN; i++)
            data_cpu[(t + 1) % 2][i] = scalar * data_cpu[t % 2][i] + 
                data_cpu[t % 2][(i - 1) % LEN] + data_cpu[t % 2][(i + 1) % LEN];
    
    /* 打印结果 */
    printf("CPU:\n");
    for (int i = 0; i < LEN - 1; i++)
        printf("%.1f\t", data_cpu[0][i]);
    printf("%.1f\n", data_cpu[0][LEN - 1]);

    /* 使用GPU计算 */
    cuda_jacobi_1d(&data_gpu[0][0], scalar, LEN, STEP);

    /* 打印结果 */
    printf("GPU:\n");
    for (int i = 0; i < LEN - 1; i++)
        printf("%.1f\t", data_gpu[0][i]);
    printf("%.1f\n", data_gpu[0][LEN - 1]);

    return 0;
}
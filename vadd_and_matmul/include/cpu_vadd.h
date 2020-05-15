/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CPU_VADD_H

#define CPU_VADD_H

//#include "intrinsic_vadd.h"
#include "configure.h"

#include <cstdlib>
#include <omp.h>

/**
 * @brief CPU向量加
 * @param c,b,a 数组
 * @param alpha,beta 标量
 * @param len 数组长度
 */
void cpu_vadd(type* c, const type* a, const type* b,
    const type& alpha, const type& beta, const int& len);

void cpu_vadd(type* c, const type* a, const type* b,
    const type& alpha, const type& beta, const int& len)
{
    /* 使用SSE向量指令集 */
    /*
#if type == float
    vadd4f(c, a, b, alpha, beta, len);
#else
#if type == double
    vadd2d(c, a, b, alpha, beta, len);
#endif
#endif
    */

    #pragma omp parallel for
    for (int i = 0; i < len; i++)
        c[i] += alpha * a[i] + beta * b[i];
}

#endif
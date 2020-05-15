/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CPU_VADD_H

#define CPU_VADD_H

/**
 * @brief CPU向量加
 * @param c,b,a 数组
 * @param alpha,beta 标量
 * @param len 数组长度
 */
template<class T>
void cpu_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& len);

template<class T>
void cpu_vadd(T* c, const T* a, const T* b,
    const T& alpha, const T& beta, const int& len)
{
    /* 使用SSE向量指令集 */
}

#endif
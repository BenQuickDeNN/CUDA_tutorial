/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CPU_MATMUL_H

#define CPU_MATMUL_H

#include "configure.h"
#include "matrix.h"

/**
 * @brief 矩阵乘
 * @param C 存储结果的矩阵
 * @param A,B 相乘的矩阵
 * @return 是否成功
 */
inline bool mat_mul(Matrix& C, const Matrix& A, const Matrix& B);

/**
 * @brief 矩阵乘重载
 * @func 只计算C矩阵中第h_start行开始到第h_end行的元素
 */
inline bool mat_mul_rows(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& h_start, const int& h_end);

/**
 * @brief 矩阵乘重载
 * @func 只计算C矩阵中第w_start列开始到第w_end列的元素
 */
inline bool mat_mul_cols(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& w_start, const int& w_end);

/**
 * @brief 矩阵乘重载
 * @func 只计算C矩阵中第h_start行开始到第h_end行，和第w_start列到第w_end列的元素
 */
inline bool mat_mul_block(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& h_start, const int& h_end, const int& w_start, const int& w_end);

inline bool mat_mul(Matrix& C, const Matrix& A, const Matrix& B)
{
    if (C.isEmpty() || A.isEmpty() || B.isEmpty())
        return false;
    if (A.getWidth() != B.getHeight())
        return false;
    if (C.getHeight() != A.getHeight())
        return false;
    if (C.getWidth() != B.getWidth())
        return false;

    for (int h = 0; h < C.getHeight(); h++)
        for (int w = 0; w < C.getWidth(); w++)
            for (int k = 0; k < A.getWidth(); k++)
                C(h, w) += A(h, k) * B(k, w);

    return true;
}

inline bool mat_mul_rows(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& h_start, const int& h_end)
{
    if (C.isEmpty() || A.isEmpty() || B.isEmpty())
        return false;
    if (A.getWidth() != B.getHeight())
        return false;
    if (C.getWidth() != B.getWidth())
        return false;
    if (h_start < 0)
        return false;
    if (h_end > C.getHeight() || h_end > A.getHeight())
        return false;

    for (int h = h_start; h < h_end; h++)
        for (int w = 0; w < C.getWidth(); w++)
            for (int k = 0; k < A.getWidth(); k++)
                C(h, w) += A(h, k) * B(k, w);

    return true;
}

inline bool mat_mul_cols(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& w_start, const int& w_end)
{
    if (C.isEmpty() || A.isEmpty() || B.isEmpty())
        return false;
    if (A.getWidth() != B.getHeight())
        return false;
    if (C.getHeight() != A.getHeight())
        return false;
    if (w_start < 0)
        return false;
    if (w_end > C.getWidth() || w_end > B.getWidth())
        return false;

    for (int h = 0; h < C.getHeight(); h++)
        for (int w = w_start; w < w_end; w++)
            for (int k = 0; k < A.getWidth(); k++)
                C(h, w) += A(h, k) * B(k, w);
    
    return true;
}

inline bool mat_mul_block(Matrix& C, const Matrix& A, const Matrix& B, 
    const int& h_start, const int& h_end, const int& w_start, const int& w_end)
{
    if (C.isEmpty() || A.isEmpty() || B.isEmpty())
        return false;
    if (A.getWidth() != B.getHeight())
        return false;
    if (h_start < 0)
        return false;
    if (h_end > C.getHeight() || h_end > A.getHeight())
        return false;
    if (w_start < 0)
        return false;
    if (w_end > C.getWidth() || w_end > B.getWidth())
        return false;
    
    for (int h = h_start; h < h_end; h++)
        for (int w = w_start; w < w_end; w++)
            for (int k = 0; k < A.getWidth(); k++)
                C(h, w) += A(h, k) * B(k, w);

    return true;
}

#endif;
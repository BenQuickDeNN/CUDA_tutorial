/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef MATRIX_H

#define MATRIX_H

#include "configure.h"

#include <cstdlib>

/**
 * @brief 矩阵类，方便矩阵计算
 */
class Matrix
{
    type* elements; // 矩阵元素
    int height = 0; // 矩阵高度
    int width = 0; // 矩阵宽度

public:
    /**
     * @brief 构造函数
     * @func 获取矩阵的尺寸信息，并申请内存空间
     * @param height 矩阵高度
     * @param width 矩阵宽度
     */
    Matrix(const int& height, const int& width);

    /**
     * @brief 析构函数
     * @func 释放内存
     */
    ~Matrix();

/**************************************************************
 * 功能函数
**************************************************************/

    /**
     * @brief 根据坐标，获取矩阵元素
     * @param h 纵向坐标
     * @param w 横向坐标
     * @return 返回矩阵元素的地址，如果矩阵为空，则返回NULL
     */
    inline type& getEle(const int& h, const int& w) const;
 
    /**
     * @brief 获取矩阵高度
     */
    inline int getHeight() const { return height; }

    /**
     * @brief 获取矩阵宽度
     */
    inline int getWidth() const { return width; }

    /**
     * @brief 判断矩阵是否为空
     */
    bool isEmpty() const { return elements == NULL; }

/***************************************************************
 * 重载运算符
***************************************************************/

    /**
     * @brief 重载括号运算符，功能与getEle函数相同
     */
    inline type& operator()(const int& h, const int& w) const;
};

/***************************************************************
 * Matrix类函数
***************************************************************/

Matrix::Matrix(const int& height, const int& width)
    : height(height), width(width)
{
    elements = (type*)std::malloc(height * width * sizeof(type));
}

Matrix::~Matrix()
{
    if (elements != NULL)
        std::free(elements);
}

inline type& Matrix::getEle (const int& h, const int& w) const
{
    if (elements == NULL)
        return *elements;
    else
        return elements[h * width + w];
}

inline type& Matrix::operator()(const int& h, const int& w) const
{ 
    return getEle(h, w); 
}

/***************************************************************
 * 非Matrix类函数
***************************************************************/

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

#endif
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

#endif
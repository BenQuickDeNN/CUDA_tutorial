/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef MATRIX_H

#define MATRIX_H

#include "configure.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>

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
    inline Matrix(const int& height, const int& width);

    /**
     * @brief 析构函数
     * @func 释放内存
     */
    inline ~Matrix();

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
    inline bool isEmpty() const { return elements == NULL; }

    /**
     * @brief 用val填满矩阵
     */
    inline bool fill(const type& val) const;

    /**
     * @brief 使用随机数填充数组
     * @param mod 随机数范围
     */
    inline bool randFill(const int& mod) const;

    /**
     * @brief 判断本矩阵与矩阵m是否相等
     * @param error 误差
     */
    inline bool isEqual(const Matrix& m, const type& error) const;

    /**
     * @brief 打印数组元素
     */
    bool disp() const;

    /**
     * @brief 打印特定行范围内的数组元素
     */
    bool disp(const int& h_start, const int& h_end) const;

/***************************************************************
 * 重载运算符
***************************************************************/

    /**
     * @brief 重载括号运算符，功能与getEle函数相同
     */
    inline type& operator()(const int& h, const int& w) const;
};

inline Matrix::Matrix(const int& height, const int& width)
    : height(height), width(width)
{
    elements = (type*)std::malloc(height * width * sizeof(type));
}

inline Matrix::~Matrix()
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

inline bool Matrix::fill(const type& val) const
{
    if (elements == NULL)
        return false;
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            getEle(h, w) = val;
    return true;
}

inline bool Matrix::randFill(const int& mod) const
{
    if (elements == NULL)
        return false;
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            getEle(h, w) = (type)(std::rand() % mod);
    return true;
}

inline bool Matrix::isEqual(const Matrix& m, const type& error) const
{
    if (elements == NULL || m.isEmpty())
        return false;
    if (height != m.getHeight())
        return false;
    if (width != m.getWidth())
        return false;

    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            if (std::fabs(getEle(h, w) - m(h, w)) > error)
            {
                fprintf(stderr, "not equal in (%d, %d)\n", h, w);
                return false;
            }
    
    return true;
}

bool Matrix::disp() const
{
    if (elements == NULL)
        return false;
    
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width - 1; w++)
            std::printf("%.1f\t", getEle(h, w));
        std::printf("%.1f\n", getEle(h, width - 1));
    }
    return true;
}

bool Matrix::disp(const int& h_start, const int& h_end) const
{
    if (elements == NULL)
        return false;
    
    for (int h = h_start; h < h_end; h++)
    {
        for (int w = 0; w < width - 1; w++)
            std::printf("%.1f\t", getEle(h, w));
        std::printf("%.1f\n", getEle(h, width - 1));
    }
    return true;
}

#endif
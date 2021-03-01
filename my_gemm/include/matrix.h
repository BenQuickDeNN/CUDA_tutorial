#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include "config.h"

/**
 * @brief 只能在host端调用的矩阵
 * 
 */
class MatirxHost
{
private:
    
public:
    type *data = NULL;
    size_t height = 0;
    size_t width = 0;

    /**
     * @brief 返回矩阵的尺寸
     * 
     * @return size_t 矩阵高度和宽度的乘积
     */
    size_t size()
    {
        return height * width;
    }

    /**
     * @brief 返回指定位置的元素
     * 
     * @param _y Y轴（高度）索引
     * @param _x X轴（宽度）索引
     * @return type& 元素的引用
     */
    type & at(const size_t &_y, const size_t &_x)
    {
        return data[_y * width + _x];
    }

    /**
     * @brief 重载括号运算符，返回指定位置的元素
     * 
     * @param _y Y轴（高度）索引
     * @param _x X轴（宽度）索引
     * @return type& 元素的引用
     */
    type & operator () (const size_t &_y, const size_t &_x)
    {
        return at(_y, _x);
    }

    /**
     * @brief 矩阵乘 _C = _A * _B
     * 
     * @param _C 储存计算结果的矩阵
     * @param _A 矩阵A
     * @param _B 矩阵B
     */
    static void Multiply(MatirxHost &_C, MatirxHost &_A, MatirxHost &_B)
    {
        using namespace std;

        if (_A.width != _B.height || _C.height != _A.height || _C.width != _B.width)
        {
            cerr << "matmul stop: host matrix cannot multiply!" << endl;
            return;
        }

#pragma omp parallel for
        for (size_t y = 0; y < _C.height; ++y)
        {
            for (size_t x = 0; x < _C.width; ++x)
            {
                type _c = 0;
                for (size_t k = 0; k < _A.width; ++k)
                {
                    _c += _A(y, k) * _B(k, x);
                }
                _C(y, x) = _c;
            }
        }
    }

    /**
     * @brief 检查矩阵是否可乘
     * 
     * @param _C 存放计算结果的矩阵
     * @param _A 矩阵A
     * @param _B 矩阵B
     * @return true 可乘
     * @return false 不可乘
     */
    static bool canMul(MatirxHost &_C, MatirxHost &_A, MatirxHost &_B)
    {
        using namespace std;
        
        if (_A.width != _B.height || _C.height != _A.height || _C.width != _B.width)
        {
            cerr << "host matrix cannot multiply!" << endl;
            return false;
        }
        return true;
    }

    /**
     * @brief 填充矩阵元素
     * 
     * @param _val 元素值
     */
    void fill(const type &_val)
    {
#pragma omp parallel for
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                at(y, x) = _val;
            }
        }
    }

    /**
     * @brief 使用随机值填充矩阵元素
     * 
     * @param _maxVal 元素最大值
     */
    void fillRandom(const size_t &_maxVal)
    {
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                at(y, x) = (type)(rand() % _maxVal);
            }
        }
    }

    /**
     * @brief 比较自身与另一个矩阵是否相同
     * 
     * @param _C 另一个矩阵
     * @param _error 可接受误差
     * @return true 相同
     * @return false 不同
     */
    bool compare(MatirxHost &_C, type _error)
    {
        using namespace std;

        if (height != _C.height || width != _C.width)
        {
            cerr << "compare fail!" << endl;
            return false;
        }

        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                if (abs(at(y, x) - _C(y, x)) > _error)
                {
                    cerr << "matrix comparation fail in position (" << y;
                    cerr << ", " << x << "), different value (";
                    cerr << at(y, x) << " and " << _C(y, x) << ")" << endl;
                    return false;
                }
            }
        }

        cout << "result pass" << endl;

        return true;
    }

    /**
     * @brief 打印显示矩阵的元素值
     * 
     */
    void display()
    {
        using namespace std;

        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                cout << at(y, x) << "\t";
            }
            cout << endl;
        }
    }

    /**
     * @brief 将矩阵元素值写入文件
     * 
     * @param _filename 文件名
     */
    void writeToFile(const std::string &_filename)
    {
        using namespace std;

        ofstream writer(_filename);
        if (!writer)
        {
            cerr << "matrix error: fail to write to file" << endl;
            return;
        }
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                writer << at(y,x);
                if (x < width - 1)
                {
                    writer << "\t";
                }
            }
            if (y < height - 1)
            {
                writer << "\r\n";
            }
        }
        writer.close();
    }

    /**
     * @brief 矩阵初始化
     * 
     * @param _sizeY 矩阵高度
     * @param _sizeX 矩阵宽度
     */
    void init(const size_t &_sizeY, const size_t &_sizeX)
    {
        using namespace std;

        data = new type[_sizeY * _sizeX];
        if (data == NULL)
        {
            cerr << "memory allocation for host matrix fail!" << endl;
        }
        height = _sizeY;
        width = _sizeX;
    }

    /**
     * @brief 回收内存
     * 
     */
    void flush()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
        }
    }

    /**
     * @brief Construct a new Matirx Host object
     * 
     * @param _sizeY 矩阵高度
     * @param _sizeX 矩阵宽度
     */
    MatirxHost(const size_t &_sizeY, const size_t &_sizeX)
    {
        init(_sizeY, _sizeX);
    }

    /**
     * @brief Destroy the Matirx Host object
     * 
     */
    ~MatirxHost()
    {
        flush();
    }
};
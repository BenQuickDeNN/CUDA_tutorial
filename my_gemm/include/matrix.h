#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include "config.h"

// 只能在host端调用的矩阵
class MatirxHost
{
private:
    
public:
    type *data = NULL;
    size_t height = 0;
    size_t width = 0;

    size_t size()
    {
        return height * width;
    }

    type & at(const size_t &_y, const size_t &_x)
    {
        return data[_y * width + _x];
    }

    type & operator () (const size_t &_y, const size_t &_x)
    {
        return at(_y, _x);
    }

    // 矩阵乘
    // _C = _A * _B
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

    // 检查矩阵是否可乘
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

    // 随机生成矩阵元素
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

    void flush()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
        }
    }

    MatirxHost(const size_t &_sizeY, const size_t &_sizeX)
    {
        init(_sizeY, _sizeX);
    }

    ~MatirxHost()
    {
        flush();
    }
};
/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef TIMER_WIN_H

#define TIMER_WIN_H
#define THOUSAND 1000.0
#define MEGA 1000000.0
#define GIGA 1000000000.0

#include <Windows.h>

/**
 * @brief Windows计时器
 */
class Timer_win
{
    LARGE_INTEGER s, e, freq;

public:
    /**
     * @brief 构造函数
     * @func 获取硬件时钟频率
     */
    Timer_win();

    /**
     * @brief 启动
     */ 
    inline void start();

    /**
     * 结束并计算经过的时间（秒）
     */ 
    inline double ends();

    /**
     * 结束并计算经过的时间（微秒）
     */ 
    inline double endus();
};

Timer_win::Timer_win() { QueryPerformanceFrequency(&freq); }

inline void Timer_win::start() { QueryPerformanceCounter(&s); }

inline double Timer_win::ends()
{
    QueryPerformanceCounter(&e);
    return (double)(e.QuadPart - s.QuadPart) / (double)freq.QuadPart;
}

inline double Timer_win::endus() { return ends() * MEGA; }

#endif
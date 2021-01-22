#pragma once

#include <cstdlib>

#define GLOBAL_GRID_SIZE 16   // CUDA grid的宽度
#define GLOBAL_BLOCK_SIZE 32  // CUDA block的宽度

typedef float type; // 设置数据类型

const size_t KILO = 1 << 10;
const size_t MEGA = 1 << 20;
const size_t GIGA = 1 << 30;

#pragma once

#include <cstdlib>

#define GLOBAL_GRID_SIZE_X 16   // CUDA grid的宽度
#define GLOBAL_GRID_SIZE_Y 16   // CUDA grid的高度
#define GLOBAL_BLOCK_SIZE_X 32  // CUDA block的宽度
#define GLOBAL_BLOCK_SIZE_Y 32  // CUDA block的高度
#define GLOBAL_WIDHT_AS 32      // shared memory 中数组As的宽度和Bs的高度

typedef float type; // 设置数据类型

const size_t KILO = 1 << 10;
const size_t MEGA = 1 << 20;
const size_t GIGA = 1 << 30;

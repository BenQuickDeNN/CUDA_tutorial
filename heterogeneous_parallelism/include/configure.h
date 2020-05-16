/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CONFIGURE_H

#define CONFIGURE_H

typedef float type; // 数据类型

const int NUM_HOST_THREAD = 3; // host端OPENMP线程数
const int NUM_THREAD = NUM_HOST_THREAD + 1;

const int NUM_SM = 16; // GPU SM的个数
const int MAX_NUM_THREAD_PER_SM = 1024; // 每个SM中允许的最大线程数

#endif
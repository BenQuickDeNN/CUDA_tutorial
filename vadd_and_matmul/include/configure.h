/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CONFIGURE_H

#define CONFIGURE_H

const int LEN = 1 << 20; // 向量加问题规模

typedef float type; // 数据类型

const int NUM_SM = 16; // GPU SM的个数
const int MAX_NUM_THREAD_PER_SM = 1024; // 每个SM中允许的最大线程数

#endif
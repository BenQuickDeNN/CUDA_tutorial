/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#ifndef CONFIGURE_H

#define CONFIGURE_H

const int LEN = 1 << 20; // 向量加问题规模

typedef float type; // 数据类型
const type alpha = 2.0, beta = 3.0; // 标量值

const int OMP_NUM_THREADS = 4; // OpenMP线程数

#endif
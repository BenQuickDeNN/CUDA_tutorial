#pragma once

#include <cuda_runtime.h>
#include "config.h"

__global__ void cuda_gemm()
{
    __shared__ As[SHARED_MEM_SIZE_Y][SHARED_MEM_SIZE_X];
    __shared__ Bs[SHARED_MEM_SIZE_Y][SHARED_MEM_SIZE_X];
}
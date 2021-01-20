#pragma once

#include "cuda_kernels.h"

bool exec_gemm_cuda_kernel(void (*_kernel)(type*, type*, type*, size_t, size_t, size_t), 
    const size_t &_device_id)
{
    return true;
}
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief cuda kernel -- compute id of a thread
 * @param array that stores thread ids
 */
__global__ void computeThreadID(unsigned int* threadID);

__global__ void computeThreadID(unsigned int* threadID)
{
    int tid = blockIdx.z * gridDim.y * gridDim.x +
        blockIdx.y * gridDim.x + blockIdx.x +
        threadIdx.z * blockDim.y * blockDim.x +
        threadIdx.y * blockDim.x + threadIdx.x;
    threadID[tid] = tid;
}
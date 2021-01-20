#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <cmath>

using namespace std;

typedef float type;
// typedef double type;
const size_t GIGA = 1 << 30;
const size_t __SIZE__ = 1 << 27; // 128M
const size_t TSTEP = 100; // 迭代步数

type C[__SIZE__], A[__SIZE__], B[__SIZE__];

__global__ void vmul(type *_C, const type *_A, const type *_B, size_t _tstep, size_t _maxIdx)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < _maxIdx; idx += stride)
    {
        type _c = 0;
        type _a = _A[idx];
        type _b = _B[idx];
        for (size_t t = 0; t < _tstep; ++t)
        {
            _c += _a * _b;
        }
        _C[idx] = _c;
    }
}

#define BLOCK_SIZE 1024
__global__ void fast_vmul(type *_C, const type *_A, const type *_B, size_t _tstep, size_t _maxIdx)
{
    __shared__ type As[BLOCK_SIZE], Bs[BLOCK_SIZE]; // 动态分配shared memory
    size_t tx = threadIdx.x;
    size_t tid = blockIdx.x * blockDim.x + tx;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < _maxIdx; idx += stride)
    {
        As[tx] = _A[idx];
        Bs[tx] = _B[idx];
        type _c = 0;
        for (size_t t = 0; t < _tstep; ++t)
        {
            _c += As[tx] * Bs[tx];
        }
        _C[idx] = _c;
    }
}

bool cuda_exec(void (*kernel)(type*, const type*, const type*, size_t, size_t), const int &_device_id)
{
    /* 选择设备 */
    if (cudaSetDevice(_device_id) != 0)
    {
        cerr << "error: GPU " << _device_id << " is not available" << endl;
        return false;
    }
    cout << "select GPU " << _device_id << endl;

    /* 设置网格grid和block */
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, _device_id);
    size_t total_workload = 3 * __SIZE__ * sizeof(type);
    std::cout << "total workload is " << (double)total_workload / (double)GIGA << " GB" << std::endl;
    if (total_workload > devProp.totalGlobalMem) // 检查工作负载是否超过内存容量
    {
        std::cerr << "error: too large total workload! Only " << (double)devProp.totalGlobalMem  / (double)GIGA << "GB are available" << std::endl;
        return false;
    }
    size_t num_threads_per_blocks = devProp.maxThreadsPerBlock;
    size_t num_blocks = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor / num_threads_per_blocks;
    cout << "num of blocks = " << num_blocks << ", ";
    cout << "num of threads per blocks = " << num_threads_per_blocks << endl;
    dim3 gridSize(num_blocks, 1, 1);
    dim3 blockSize(num_threads_per_blocks, 1, 1);

    /* 计算每个block对应的shared memory容量 */
    cout << "the shared memory size per block is " << devProp.sharedMemPerBlock / 1024 << " KB" << endl;

    /* 分配内存 */
    type *cu_C, *cu_A, *cu_B;
    cudaMalloc((void**)&cu_C, __SIZE__ * sizeof(type));
    cudaMalloc((void**)&cu_A, __SIZE__ * sizeof(type));
    cudaMalloc((void**)&cu_B, __SIZE__ * sizeof(type));

    /* GPU热身 */
    kernel<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, TSTEP, __SIZE__);

    /* 传输数据 */
    cudaMemcpy(cu_A, A, __SIZE__ * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_B, B, __SIZE__ * sizeof(type), cudaMemcpyHostToDevice);

    /* 执行内核 */
    cout << "start kernel..." << endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start, 0);

    kernel<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, TSTEP, __SIZE__);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "kernel elapsed " << elapsed << " ms, ";
    cout << "the speed is " << (float)(__SIZE__ * 2 * TSTEP) * 1000 / elapsed / (float)GIGA << " GFlops" << endl;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    /* 传回结果 */
    cudaMemcpy(C, cu_C, __SIZE__ * sizeof(type), cudaMemcpyDeviceToHost);

    /* 释放内存 */
    cudaFree(cu_B);
    cudaFree(cu_A);
    cudaFree(cu_C);

    return true;
}

int main()
{
    if (cuda_exec(fast_vmul, 1))
    {
        cout << "computation finished" << endl;
    }
    else
    {
        cerr << "computation fail!" << endl;
    }
    return 0;
}
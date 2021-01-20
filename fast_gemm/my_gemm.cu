#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>

using namespace std;

const size_t GIGA = 1 << 30;

typedef float type;
// typedef double type;

const size_t WidthA = 1024;
const size_t HeightA = 1024;
const size_t SizeA = HeightA * WidthA;

const size_t WidthB = 1024;
const size_t HeightB = 1024;
const size_t SizeB = HeightB * WidthB;

const size_t SizeC = HeightA * WidthB;

type A[SizeA], B[SizeB], C[SizeC];

__global__ void cuda_gemm(type *_C, type *_A, type *_B, size_t _wC, size_t _wA, size_t _wB, size_t _maxIdx)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 线程号
    size_t stride = gridDim.x * blockDim.x; // 索引更新步长
    for (size_t idx = tid; idx < _maxIdx; idx += stride)
    {
        size_t y = idx / _wC;
        size_t x = idx % _wC;
        type _c = 0;
        size_t y_wA = y * _wA;
        type *p_A = _A + y_wA;
        /* 计算 */
// #pragma unroll
        for (size_t k = 0; k < _wA; ++k)
        {
            _c += p_A[k] * _B[k * _wB + x];
        }
        _C[idx] = _c;
    }
}

// 使用shared memory
__global__ void cuda_gemm2(type *_C, type *_A, type *_B, size_t _wC, size_t _wA, size_t _wB, size_t _maxIdx, 
    size_t _blockSize, size_t _numBlocks)
{
    /* 对矩阵做分块 */
    extern __shared__ type As[], Bs[]; // 动态分配
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // 线程号
    size_t stride = gridDim.x * blockDim.x; // 索引更新步长
    for (size_t idx = tid; idx < _maxIdx; idx += stride)
    {
        size_t y = idx / _wC;
        size_t x = idx % _wC;
        type _c = 0;
        size_t y_wA = y * _wA;
        /* 计算 */
        for (size_t k = 0; k < _wA; ++k)
        {
            _c += _A[y_wA + k] * _B[k * _wB + x];
        }
        _C[idx] = _c;
    }
}

bool cuda_exec_gemm(const int &_device_id)
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
    size_t total_workload = (SizeC + SizeA + SizeB) * sizeof(type);
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
    size_t sharedMemSize = devProp.sharedMemPerBlock / sizeof(type) / 2;
    cout << "the shared memory size per block is " << devProp.sharedMemPerBlock / 1024 << " KB" << endl;
    size_t _numBlocks = (size_t)ceil((double)SizeC / (double)sharedMemSize);

    /* 分配内存 */
    type *cu_C, *cu_A, *cu_B;
    cudaMalloc((void**)&cu_C, SizeC * sizeof(type));
    cudaMalloc((void**)&cu_A, SizeA * sizeof(type));
    cudaMalloc((void**)&cu_B, SizeB * sizeof(type));

    /* GPU热身 */
    cuda_gemm<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, WidthB, WidthA, WidthB, SizeC);

    /* 传输数据 */
    cudaMemcpy(cu_A, A, SizeA * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_B, B, SizeB * sizeof(type), cudaMemcpyHostToDevice);

    /* 执行内核 */
    cout << "start kernel..." << endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start, 0);

    // cuda_gemm<<<gridSize, blockSize>>>(cu_C, cu_A, cu_B, WidthB, WidthA, WidthB, SizeC);
    cuda_gemm2<<<gridSize, blockSize, sharedMemSize>>>(cu_C, cu_A, cu_B, WidthB, WidthA, WidthB, SizeC, sharedMemSize, _numBlocks);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "kernel elapsed " << elapsed << " ms, ";
    cout << "the speed is " << (float)(SizeC * 2 * WidthA) * 1000 / elapsed / (float)GIGA << " GFlops" << endl;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    /* 传回结果 */
    cudaMemcpy(C, cu_C, SizeC * sizeof(type), cudaMemcpyDeviceToHost);

    /* 释放内存 */
    cudaFree(cu_B);
    cudaFree(cu_A);
    cudaFree(cu_C);

    return true;
}

void myfill(type *arr, const size_t &_height, const size_t &_width, const type &_val)
{
#pragma omp parallel for
    for (size_t h = 0; h < _height; ++h)
    {
        const size_t idx1 = h * _width;
        for (size_t w = 0; w < _width; ++w)
        {
            arr[idx1 + w] = _val;
        }
    }
}

void verify(const type *cu_C, const type *A, const type *B)
{
    type *C = new type[HeightA * WidthB];
#pragma omp parallel for
    for (size_t h = 0; h < HeightA; ++h)
    {
        const size_t idx1 = h * WidthB;
        for (size_t w = 0; w < WidthB; ++w)
        {
            const size_t idx2 = idx1 + w;
            C[idx2] = 0.0;
#pragma unroll
            for (size_t k = 0; k < WidthA; ++k)
            {
                C[idx2] += A[idx1 + k] * B[k * WidthB + w];
            }
        }
    }
    for (size_t h = 0; h < HeightA; ++h)
    {
        const size_t idx1 = h * WidthB;
        for (size_t w = 0; w < WidthB; ++w)
        {
            const size_t idx2 = idx1 + w;
            if (abs(C[idx2] - cu_C[idx2]) > 0.1)
            {
                cerr << "computation error occurs where h = " << h << ", w = " << w;
                cerr <<"; and C = " << C[idx2] << " but cu_C = " << cu_C[idx2] << endl;
                return;
            }
        }
    }
    cout << "computation correct" << endl;
}

int main()
{
    myfill(A, HeightA, WidthA, 2.0);
    myfill(B, HeightB, WidthB, 3.0);
    if (cuda_exec_gemm(1))
    {
        verify(C, A, B);
    }
    else
    {
        cerr << "computation fail!" << endl;
    }
    return 0;
}
#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include "config.h"
#include "cuda_kernels.h"
#include "matrix.h"

class CUDAHelper
{
public:
    static bool setGPU(const size_t &_deviceId)
    {
        using namespace std;

        if (auto code = cudaSetDevice(_deviceId) != cudaSuccess)
        {
            cerr << "error: fail to set device " << _deviceId << ", error code = " << code << endl;
            return false;
        }

        return true;
    }

    static bool getGPUInfo(const size_t &_deviceId, std::string *_name, 
        size_t *_maxThreadsPerBlock, size_t *_maxBlocksPerSM, size_t *_smCount,
        size_t *_totalGlobalMem, size_t *_sharedMemPerBlock)
    {
        using namespace std;

        cudaDeviceProp devProp;
        if (auto code = cudaGetDeviceProperties(&devProp, _deviceId) != cudaSuccess)
        {
            cerr << "fail to fetch GPU infomation, error code = " << code << endl;
            return false;
        }

        *_name = devProp.name;
        *_maxThreadsPerBlock = devProp.maxThreadsPerBlock;
        *_maxBlocksPerSM = devProp.maxBlocksPerMultiProcessor;
        *_smCount = devProp.multiProcessorCount;
        *_totalGlobalMem = devProp.totalGlobalMem;
        *_sharedMemPerBlock = devProp.sharedMemPerBlock;

        return true;
    }

    static void printGPUInfo(const size_t &_deviceId, const std::string &_name,
        const size_t &_maxThreadsPerBlock, const size_t &_maxBlocksPerSM, const size_t &_smCount,
        const size_t &_totalGlobalMem, const size_t &_sharedMemPerBlock)
    {
        using namespace std;

        cout << "Device Id: " << _deviceId << endl;
        cout << "GPU Name: " << _name << endl;
        cout << "Max Threads Per Block: " << _maxThreadsPerBlock << endl;
        cout << "Max blocks Per SM: " << _maxBlocksPerSM << endl;
        cout << "SM Count: " << _smCount << endl;
        cout << "Total Global Memory: " << (double)_totalGlobalMem / (double)GIGA << " GB" << endl;
        cout << "Shared Memory Per Block: " << (double)_sharedMemPerBlock / (double)KILO << " KB" << endl;
    }

    static bool cuMalloc(void **_addr, const size_t &_length)
    {
        using namespace std;

        if (auto code = cudaMalloc(_addr, _length * sizeof(type)) != cudaSuccess)
        {
            cerr << "error: memory allocation on GPU fail, error code = " << code << endl;
            return false;
        }

        return true;
    }

    static bool cuFree(type *_arr)
    {
        using namespace std;

        if (_arr != NULL)
        {
            if (auto code = cudaFree(_arr) != cudaSuccess)
            {
                cerr << "error: memory free on GPU fail, error code = " << code << endl;
                return false;
            }
            _arr = NULL;
        }

        return true;
    }

    static bool sendData(type *_source, type *_target, const size_t &_length)
    {
        using namespace std;

        if (auto code = cudaMemcpy(_target, _source, _length * sizeof(type), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cerr << "error: sending data fail, error code = " << code << endl;
            return false;
        }

        return true;
    }

    static bool receivedData(type *_source, type *_target, const size_t &_length)
    {
        using namespace std;

        if (auto code = cudaMemcpy(_target, _source, _length * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cerr << "error: receiving data fail, error code = " << code << endl;
            return false;
        }

        return true;
    }
};

template <size_t GRID_SIZE, size_t BLOCK_SIZE>
bool exec_cuda_gemm_kernel(MatirxHost &_C, MatirxHost &_A, MatirxHost &_B,
    const bool &_flag_sharedmem,
    const size_t &_deviceId = 0)
{
    using namespace std;

    // 检查矩阵是否可乘
    if (!MatirxHost::canMul(_C, _A, _B)) { return false; }

    // 选择和检测GPU
    if (!CUDAHelper::setGPU(_deviceId)) { return false; }

    // 获取GPU属性
    string gpuinfo_name;
    size_t gpuinfo_max_threads_per_block;
    size_t gpuinfo_max_blocks_per_sm;
    size_t gpuinfo_sm_count;
    size_t gpuinfo_total_global_mem;
    size_t gpuinfo_shared_mem_per_block;
    if (!CUDAHelper::getGPUInfo(_deviceId, &gpuinfo_name, 
        &gpuinfo_max_threads_per_block, &gpuinfo_max_blocks_per_sm, &gpuinfo_sm_count,
        &gpuinfo_total_global_mem, &gpuinfo_shared_mem_per_block))
    { return false; }

    // 打印GPU属性
    CUDAHelper::printGPUInfo(_deviceId, gpuinfo_name,
        gpuinfo_max_threads_per_block, gpuinfo_max_blocks_per_sm, gpuinfo_sm_count,
        gpuinfo_total_global_mem, gpuinfo_shared_mem_per_block);

    // 打印CUDA配置
    cout << "GRID_SIZE = " << GRID_SIZE << endl;
    cout << "BLOCK_SIZE = " << BLOCK_SIZE << endl;
    cout << "USE_SHARED_MEM = ";
    if (_flag_sharedmem)
    {
        cout << "true";
    }
    else
    {
        cout << "false";
    }
    cout << endl;

    // 检查网格大小是否符合要求
    if (GRID_SIZE * GRID_SIZE > gpuinfo_max_blocks_per_sm * gpuinfo_sm_count)
    {
        cerr << "stop: too large grid size" << endl;
        return false;
    }

    // 检查block大小是否符合要求
    if (BLOCK_SIZE * BLOCK_SIZE > gpuinfo_max_threads_per_block)
    {
        cerr << "stop: too larege block size" << endl;
        return false;
    }

    // 打印数据精度
    cout << "Precise: ";
    if (sizeof(type) == sizeof(float))
    {
        cout << "single";
    }
    else if (sizeof(type) == sizeof(double))
    {
        cout << "double";
    }
    else
    {
        cout << sizeof(type) << " bytes";
    }
    cout << endl;

    // 检查内存容量是否满足计算要求
    if ((_C.size() + _A.size() + _B.size()) * sizeof(type) >= gpuinfo_total_global_mem)
    {
        cerr << "stop: data size is larger than global memory capacity" << endl;
        return false;
    }

    // 检查分配到shared memory的数据规模是否超过shared memory容量
    if (BLOCK_SIZE * BLOCK_SIZE * 2 * sizeof(type) >= gpuinfo_shared_mem_per_block)
    {
        cerr << "stop: data size on shared memory is larger than shared memory capacity" << endl;
        return false;
    }

    // 设置网格grid和block
    dim3 grid_dim(GRID_SIZE, GRID_SIZE, 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // 分配内存
    type *cu_C = NULL, *cu_A = NULL, *cu_B = NULL;
    if (!CUDAHelper::cuMalloc(reinterpret_cast<void **>(&cu_C), _C.size())) { return false; }
    if (!CUDAHelper::cuMalloc(reinterpret_cast<void **>(&cu_A), _A.size())) { return false; }
    if (!CUDAHelper::cuMalloc(reinterpret_cast<void **>(&cu_B), _B.size())) { return false; }

    // 传输数据
    // bug: 数据可能没传上
    if (!CUDAHelper::sendData(_A.data, cu_A, _A.size())) { return false; }
    if (!CUDAHelper::sendData(_B.data, cu_B, _B.size())) { return false; }

    // GPU热身
    if (_flag_sharedmem)
    {
        cuda_gemm_shared_mem<BLOCK_SIZE><<<grid_dim, block_dim>>>(cu_C, cu_A, cu_B,
            _C.height, _C.width, _A.width, _B.width, 
            0, 0);
    }
    else
    {
        cuda_gemm<<<grid_dim, block_dim>>>(cu_C, cu_A, cu_B, 
                    _C.height, _C.width, _A.width, _B.width, 
                    0, 0);
    }
    
    cudaDeviceSynchronize();

    // 执行内核
    cout << "start kernel..." << endl;

    const size_t grid_length_X = GRID_SIZE * BLOCK_SIZE;
    const size_t grid_length_Y = GRID_SIZE * BLOCK_SIZE;
    const size_t blocks_X = (size_t)ceil((double)_C.width / (double)grid_length_X);
    const size_t blocks_Y = (size_t)ceil((double)_C.height / (double)grid_length_Y);

    cudaEvent_t start, stop;
    float elapsed;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    if (_flag_sharedmem)
    {
        for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y)
        {
            for (size_t block_X = 0; block_X < blocks_X; ++block_X)
            {
                cuda_gemm_shared_mem<BLOCK_SIZE><<<grid_dim, block_dim>>>(cu_C, cu_A, cu_B,
                    _C.height, _C.width, _A.width, _B.width, 
                    block_Y * grid_length_Y, block_X * grid_length_X);
                cudaDeviceSynchronize();
            }
        }
    }
    else
    {
        for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y)
        {
            for (size_t block_X = 0; block_X < blocks_X; ++block_X)
            {
                cuda_gemm<<<grid_dim, block_dim>>>(cu_C, cu_A, cu_B, 
                    _C.height, _C.width, _A.width, _B.width, 
                    block_Y * grid_length_Y, block_X * grid_length_X);
                cudaDeviceSynchronize();
            }
        }
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cout << "kernel elapsed " << elapsed << " ms" << endl;
    cout << "computation speed is " << (float)(_C.size() * _A.width * 2 * 1000) / elapsed / (float)GIGA << " GFlops" << endl;

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    // 传回数据
    if (!CUDAHelper::receivedData(cu_C, _C.data, _C.size())) { return false; }
    
    // 释放内存
    if (!CUDAHelper::cuFree(cu_B)) { return false; }
    if (!CUDAHelper::cuFree(cu_A)) { return false; }
    if (!CUDAHelper::cuFree(cu_C)) { return false; }

    return true;
}
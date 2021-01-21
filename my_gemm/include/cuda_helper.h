#pragma once

#include <iostream>
#include <string>
#include "config.h"
#include "cuda_kernels.h"
#include "matrix.h"

class CUDAHelper
{
public:
    static bool setGPU(const size_t &_deviceId)
    {
        using namespace std;

        if (cudaSetDevice(_deviceId) != 0)
        {
            cerr << "error: device " << _deviceId << " is not available" << endl;
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
        if (cudaGetDeviceProperties(&devProp, _deviceId) != 0)
        {
            cerr << "fail to fetch GPU infomation!" << endl;
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
        cout << "Max Blocks Per SM: " << _maxBlocksPerSM << endl;
        cout << "SM Count: " << _smCount << endl;
        cout << "Total Global Memory: " << (double)_totalGlobalMem / (double)GIGA << " GB" << endl;
        cout << "Shared Memory Per Block: " << (double)_sharedMemPerBlock / (double)KILO << " KB" << endl;
    }

    static bool cuMalloc(type *_arr, const size_t &_length)
    {
        using namespace std;

        if (cudaMalloc((void**)&_arr, _length * sizeof(type)) != 0)
        {
            cerr << "error: memory allocation on GPU fail" << endl;
            return false;
        }

        return true;
    }

    static bool cuFree(type *_arr)
    {
        using namespace std;

        if (_arr != NULL)
        {
            if (cudaFree(_arr) != 0)
            {
                cerr << "error: memory free on GPU fail" << endl;
                return false;
            }
            _arr = NULL;
        }

        return true;
    }

    static bool sendData(type *_source, type *_target, const size_t &_length)
    {
        using namespace std;

        if (!cudaMemcpy(_target, _source, _length * sizeof(type), cudaMemcpyHostToDevice) != 0)
        {
            cerr << "error: sending data fail" << endl;
            return false;
        }

        return true;
    }

    static bool receivedData(type *_source, type *_target, const size_t &_length)
    {
        using namespace std;

        if (!cudaMemcpy(_target, _source, _length * sizeof(type), cudaMemcpyDeviceToHost) != 0)
        {
            cerr << "error: receiving data fail" << endl;
            return false;
        }

        return true;
    }
};

bool exec_cuda_gemm_kernel(MatirxHost &_C, MatirxHost &_A, MatirxHost &_B,
    const size_t &_gridSizeX, const size_t &_gridSizeY, 
    const size_t &_blockSizeX, const size_t &_blockSizeY,
    const size_t &_widthAs,
    const size_t &_deviceId = 0)
{
    using namespace std;

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

    // 检查网格大小是否符合要求
    if (_gridSizeX * _gridSizeY > gpuinfo_max_blocks_per_sm * gpuinfo_sm_count)
    {
        cerr << "stop: too large grid size" << endl;
        return false;
    }

    // 检查block大小是否符合要求
    if (_blockSizeX * _blockSizeY > gpuinfo_max_threads_per_block)
    {
        cerr << "stop: too larege block size" << endl;
        return false;
    }

    // 检查内存容量是否满足计算要求
    if ((_C.size() + _A.size() + _B.size()) * sizeof(type) >= gpuinfo_total_global_mem)
    {
        cerr << "stop: data size is larger than global memory capacity" << endl;
        return false;
    }

    // 检查分配到shared memory的数据规模是否超过shared memory容量
    if ((_blockSizeX + _blockSizeY) * _widthAs * sizeof(type) >= gpuinfo_shared_mem_per_block)
    {
        cerr << "stop: data size on shared memory is larger than shared memory capacity" << endl;
        return false;
    }

    // 设置网格grid和block
    dim3 grid_dim(_gridSizeX, _gridSizeY, 1);
    dim3 block_dim(_blockSizeX, _blockSizeY, 1);

    // 分配内存
    type *cu_C, *cu_A, *cu_B;
    if (!CUDAHelper::cuMalloc(cu_C, _C.size())) { return false; }
    if (!CUDAHelper::cuMalloc(cu_A, _A.size())) { return false; }
    if (!CUDAHelper::cuMalloc(cu_B, _B.size())) { return false; }

    // 传输数据
    if (!CUDAHelper::sendData(_A.getArray(), cu_A, _A.size())) { return false; }
    if (!CUDAHelper::sendData(_B.getArray(), cu_B, _B.size())) { return false; }

    // 执行内核

    // 传回数据
    if (!CUDAHelper::receivedData(cu_C, _C.getArray(), _C.size())) { return false; }
    
    // 释放内存
    if (!CUDAHelper::cuFree(cu_B)) { return false; }
    if (!CUDAHelper::cuFree(cu_A)) { return false; }
    if (!CUDAHelper::cuFree(cu_C)) { return false; }

    return true;
}
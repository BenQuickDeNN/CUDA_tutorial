#include <iostream>
#include <omp.h>
#include <cmath>

#include "cuda_helper.cu"

using namespace std;

const size_t height_C = 80 * 32 * 1024 * 8;
const size_t width_C = 64;
const size_t width_A = 64;
const size_t workload_h_gpu0 = height_C / 2;
const size_t workload_h_gpu1 = height_C - workload_h_gpu0;

void myfill(type *arr, const size_t &_height, const size_t &_width, const type &_val);
void verify(const type *cu_C, const type *A, const type *B);

int main()
{
    type *C, *A, *B;
    C = new type[height_C * width_C];
    A = new type[height_C * width_A];
    B = new type[width_A * width_C];

/*
#pragma omp parallel for num_threads(2)
    for (size_t device = 0; device < 2; ++device)
    {
        cuda_exec_gemm(C + workload_h_per_gpu * device * width_C, A, B, height_C, width_C, width_A, device);
    }
*/
    cuda_exec_gemm(C + workload_h_per_gpu * 0 * width_C, A, B, height_C, width_C, width_A, 0);

    delete[] B;
    delete[] A;
    delete[] C;
    return 0;
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
    type *C = new type[height_C * width_C];
#pragma omp parallel for
    for (size_t h = 0; h < height_C; ++h)
    {
        const size_t idx1 = h * width_C;
        for (size_t w = 0; w < width_C; ++w)
        {
            const size_t idx2 = idx1 + w;
            C[idx2] = 0.0;
            for (size_t k = 0; k < width_A; ++k)
            {
                C[idx2] += A[idx1 + k] * B[k * width_C + w];
            }
        }
    }
    for (size_t h = 0; h < height_C; ++h)
    {
        const size_t idx1 = h * width_C;
        for (size_t w = 0; w < width_C; ++w)
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
#include <iostream>
#include <omp.h>

#include "cuda_helper.cu"

using namespace std;

const size_t height_C = 4096;
const size_t width_C = 4096;
const size_t width_A = 4096;
const size_t workload_h_per_gpu = 2048;

int main()
{
    type *C, *A, *B;
    C = new type[height_C * width_C];
    A = new type[height_C * width_A];
    B = new type[width_A * width_C];

#pragma omp parallel for num_threads(2)
    for (size_t device = 0; device < 2; ++device)
    {
        cuda_exec_gemm(C, A, B, height_C, width_C, workload_h_per_gpu * device, width_A, device);
    }

    delete[] B;
    delete[] A;
    delete[] C;
    return 0;
}
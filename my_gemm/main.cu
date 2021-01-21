#include <cstdlib>
#include "config.h"
#include "matrix.h"
#include "cuda_helper.h"

using namespace std;

const size_t HeightA = 1024, WidthA = 1024;
const size_t HeightB = WidthA, WidthB = 1024;
const size_t HeightC = HeightA, WidthC = WidthB;

int main()
{
    MatirxHost A(HeightA, WidthA), B(HeightB, WidthB);
    MatirxHost C(HeightC, WidthC), C_verify(HeightC, WidthC);

    A.fill(2.0); B.fill(3.0);

    exec_cuda_gemm_kernel(C, A, B, 16, 16, 32, 32, 32, 1);

    MatirxHost::Multiply(C_verify, A, B);

    return 0;
}
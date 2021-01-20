#include <cstdlib>
#include "config.h"
#include "matrix.h"
#include "cuda_helpers.h"

using namespace std;

const size_t HeightA = 1024, WidthA = 1024, SizeA = HeightA * WidthA;
const size_t HeightB = WidthA, WidthB = 1024, SizeB = HeightB * WidthB;
const size_t HeightC = HeightA, WidthC = WidthB, SizeC = HeightC * WidthC;

int main()
{
    MatirxHost A(HeightA, WidthA), B(HeightB, WidthB);
    MatirxHost C(HeightC, WidthC), C_verify(HeightC, WidthC);

    A.fill(2.0); B.fill(3.0);

    exec_gemm_cuda_kernel(cuda_gemm, 1);

    MatirxHost::Multiply(C_verify, A, B);

    return 0;
}
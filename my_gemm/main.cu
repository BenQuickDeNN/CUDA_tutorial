#include <cstdlib>
#include "config.h"
#include "matrix.h"
#include "cuda_helper.h"

using namespace std;

const size_t HeightA = 500, WidthA = 600;
const size_t HeightB = WidthA, WidthB = 700;
const size_t HeightC = HeightA, WidthC = WidthB;

int main()
{
    MatirxHost A(HeightA, WidthA), B(HeightB, WidthB);
    MatirxHost C(HeightC, WidthC), C_verify(HeightC, WidthC);

    A.fill(2.0); B.fill(3.0);

    // 执行CUDA内核，根据GPU计算能力和矩阵形状选择适合的grid和block
    if (exec_cuda_gemm_kernel<12, 12, 32, 32, 32>(C, A, B, 1))
    {
        MatirxHost::Multiply(C_verify, A, B);
        C_verify.compare(C, 0.1);

        // cout << "C:" << endl;
        // C.display();
        // cout << endl << "C_verify:" << endl;
        // C_verify.display();
    }

    return 0;
}
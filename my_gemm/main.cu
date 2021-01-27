#include <cstdlib>
#include "config.h"
#include "matrix.h"
#include "cuda_helper.h"
#include "cli.h"

using namespace std;

int main(int argc, char **argv)
{
    size_t HeightA = GLOBAL_SIZE, WidthA = GLOBAL_SIZE;
    size_t HeightB = WidthA, WidthB = GLOBAL_SIZE;

    size_t device_id = 0;

    bool flag_continue = true;
    bool flag_sharedmem = false;

    // 解析命令行参数
    while (true)
    {
        int option_index = 0;
        int c = getopt_long(argc, argv, "hvdsgktwa", CLI_LONG_OPTIONS, &option_index);

        // 如果没有命令行参数
        if (c == -1)
        {
            break;
        }

        switch (c)
        {
        case 'h':
            showHelpInfo();
            flag_continue = false;
            break;
        case 'v':
            showVersionInfo();
            flag_continue = false;
            break;
        case 'd':
            device_id = atoi(optarg);
            break;
        case 's':
            flag_sharedmem = true;
            break;
        case 'g':
            HeightA = atoi(optarg);
            break;
        case 'k':
            WidthA = atoi(optarg);
            break;
        case 't':
            HeightB = atoi(optarg);
            break;
        case 'w':
            WidthB = atoi(optarg);
            break;
        case 'a':
            HeightA = atoi(optarg);
            WidthA = HeightA;
            HeightB = HeightA;
            WidthB = HeightA;
            break;
        default:
            showInvalidCLIInfo();
            flag_continue = false;
            break;
        }
    }

    if (!flag_continue)
    {
        exit(0);
    }

    cout << "HeightA = " << HeightA << ", WidhtA = " << WidthA << endl;
    cout << "HeightB = " << HeightB << ", WidthB = " << WidthB << endl;

    size_t HeightC = HeightA, WidthC = WidthB;

    MatirxHost A(HeightA, WidthA), B(HeightB, WidthB);
    MatirxHost C(HeightC, WidthC), C_verify(HeightC, WidthC);

    // A.fill(2.0); B.fill(3.0);
    A.fillRandom(10); B.fillRandom(10);
    A.writeToFile("matrix_A.txt");
    B.writeToFile("matrix_B.txt");

    // 执行CUDA内核，根据GPU计算能力和矩阵形状选择适合的grid和block
    if (exec_cuda_gemm_kernel<GLOBAL_GRID_SIZE, GLOBAL_BLOCK_SIZE>(C, A, B, flag_sharedmem, device_id))
    {
        MatirxHost::Multiply(C_verify, A, B);
        C_verify.compare(C, 0.1);

        // cout << "C:" << endl;
        // C.display();
        // cout << endl << "C_verify:" << endl;
        // C_verify.display();

        C.writeToFile("matrix_C.txt");
        C_verify.writeToFile("matrix_C_verify.txt");
    }

    exit(0);
}
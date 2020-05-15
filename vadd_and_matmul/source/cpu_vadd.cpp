/*********************************************************************
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
*********************************************************************/

#include "configure.h"
#include "cpu_vadd.h"
#include "timer_win.h"

#include <cstdio>

type a[LEN], b[LEN], c[LEN];
const type alpha = 2.0;
const type beta = 3.0;

int main()
{
    using namespace std;

    /* 执行CPU数组加 */
    Timer_win tw;
    tw.start();
    cpu_vadd(c, a, b, alpha, beta, LEN);
    double elapsed = tw.endus();

    /* 打印运行时间 */
    printf("CPU arrayadd elapsed = %.3f us\r\n", elapsed);

    /* 浮点计算速率 */
    double speed = (double)LEN / elapsed / THOUSAND;
    if (sizeof(type) == sizeof(float))
        fprintf(stdout, "the speed of CPU arrayadd is %.3f GFLOPS \r\n", speed);
    else if (sizeof(type) == sizeof(double))
        fprintf(stdout, "the speed of CPU arrayadd is %.3f GDFLOPS \r\n", speed);

    return 0;
}
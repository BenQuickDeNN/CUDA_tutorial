/**********************************************
 * @brief vector add implementation
 * @author Bin Qu
 * @email benquickdenn@foxmail.com
**********************************************/

#ifndef INTRINSIC_VADD_H

#define INTRINSIC_VADD_H

#include <intrin.h>
#include <omp.h>

/**
 * @brief vector add for 128 bits float
 * @param c array to store
 * @param a,b arrays to read
 * @param len length of array
 */
inline void vadd4f(float* c, const float* a, const float* b, 
    const float& alpha, const float& beta, const int& len);

/**
 * @brief vector add for 128 bits double
 * @param c array to store
 * @param a,b arrays to read
 * @param len length of array
 */
inline void vadd2d(double* c, const double* a, const double* b, 
    const double& alpha, const double& beta, const int& len);

inline void vadd4f(float* c, const float* a, const float* b, 
    const float& alpha, const float& beta, const int& len)
{
    const unsigned long len1 = (len / 4) * 4; // length of aligned elements
    const unsigned long len2 = len % 4;       // length of unaligned elements
    __m128 xmm_a, xmm_b, xmm_c, xmm_alpha, xmm_beta;
    xmm_alpha = _mm_set1_ps(alpha);
    xmm_beta = _mm_set1_ps(beta);
    #pragma omp parallel for private(xmm_a, xmm_b, xmm_c, xmm_alpha, xmm_beta)
    for (unsigned long i = 0; i < len1; i += 4)
    {
        xmm_a = _mm_load_ps(a + i);
        xmm_b = _mm_load_ps(b + i);
        xmm_c = _mm_load_ps(c + i);
        xmm_c = _mm_add_ps(xmm_c, 
            _mm_add_ps(_mm_add_ps(xmm_alpha, xmm_a), _mm_add_ps(xmm_beta, xmm_b)));
        _mm_store_ps(c + i, xmm_c);
    }
    for (unsigned int i = len1; i < len2; i++)
        c[i] = a[i] + b[i];
}

inline void vadd2d(double* c, const double* a, const double* b, 
    const double& alpha, const double& beta, const int& len)
{
    const unsigned long len1 = (len / 2) * 2; // length of aligned elements
    const unsigned long len2 = len % 2;       // length of unaligned elements
    __m128d xmm_a, xmm_b, xmm_c, xmm_alpha, xmm_beta;
    xmm_alpha = _mm_set1_pd(alpha);
    xmm_beta = _mm_set1_pd(beta);
    #pragma omp parallel for private(xmm_a, xmm_b, xmm_c, xmm_alpha, xmm_beta)
    for (unsigned long i = 0; i < len1; i += 2)
    {
        xmm_a = _mm_load_pd(a + i);
        xmm_b = _mm_load_pd(b + i);
        xmm_c = _mm_load_pd(c + i);
        xmm_c = _mm_add_pd(xmm_c, 
            _mm_add_pd(_mm_add_pd(xmm_alpha, xmm_a), _mm_add_pd(xmm_beta, xmm_b)));
        _mm_store_pd(c + i, xmm_c);
    }
    for (unsigned int i = len1; i < len2; i++)
        c[i] = a[i] + b[i];
}

#endif
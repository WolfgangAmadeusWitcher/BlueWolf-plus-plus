#ifndef BWPP_CPU_REF_H
#define BWPP_CPU_REF_H

#include <stdint.h>

void bwpp_cpu_matmul_f32(const float *a,
                         const float *b,
                         float *c,
                         uint32_t M,
                         uint32_t N,
                         uint32_t K,
                         uint32_t lda,
                         uint32_t ldb,
                         uint32_t ldc,
                         const float *bias,
                         int apply_silu,
                         int apply_bias);

void bwpp_cpu_softmax_f32(const float *x,
                          float *y,
                          uint32_t rows,
                          uint32_t cols,
                          uint32_t ld);

void bwpp_cpu_rmsnorm_f32(const float *x,
                          float *y,
                          const float *gamma,
                          const float *beta,
                          uint32_t rows,
                          uint32_t cols,
                          uint32_t ld,
                          float eps);

#endif

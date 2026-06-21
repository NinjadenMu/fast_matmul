#ifndef MATMUL_H
#define MATMUL_H

typedef void (*matmul_func_t)(int n, const float *restrict A,
                              const float *restrict B, float *restrict C);

void matmul_naive(int n, const float *restrict A, const float *restrict B,
                  float *restrict C);
void matmul_permuted(int n, const float *restrict A, const float *restrict B,
                     float *restrict C);
void matmul_tiled(int n, const float *restrict A, const float *restrict B,
                  float *restrict C);
void matmul_micro_kernel(int n, const float *restrict A,
                         const float *restrict B, float *restrict C);
void matmul_vectorized(int n, const float *restrict A, const float *restrict B,
                       float *restrict C);

#endif

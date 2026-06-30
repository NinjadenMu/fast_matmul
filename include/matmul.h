#ifndef MATMUL_H
#define MATMUL_H

#define MEM_ALIGNMENT_MIN 128

typedef int (*matmul_func_t)(int n, const float *restrict A,
                             const float *restrict B, float *restrict C);

int matmul_naive(int n, const float *restrict A, const float *restrict B,
                 float *restrict C);
int matmul_permuted(int n, const float *restrict A, const float *restrict B,
                    float *restrict C);
int matmul_tiled(int n, const float *restrict A, const float *restrict B,
                 float *restrict C);
int matmul_micro_kernel(int n, const float *restrict A, const float *restrict B,
                        float *restrict C);
int matmul_vectorized(int n, const float *restrict A, const float *restrict B,
                      float *restrict C);
int matmul_blis(int n, const float *restrict A, const float *restrict B,
                float *restrict C);
int matmul_parallel(int n, const float *restrict A, const float *restrict B,
                float *restrict C);

int get_env_int(const char *env_var, int default_val);

#endif

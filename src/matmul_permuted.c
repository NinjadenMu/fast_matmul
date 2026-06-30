/**
 * @file matmul_permuted.c
 * @brief a faster matmul implementation through a simple locality optimization
 *
 * This file assumes all matrices are in column-major order, and that the
 * output buffer is 0-initialized.  By permuting the loops such that the
 * innermost index is the one with the most spatially local access pattern
 * (just walking down a column), cache lines may be used multiple times,
 * greatly improving cache utilization.
 *
 * It's still intended to be a simple baseline implementation, so it
 * should be compiled with vectorization optimizations off.
 */

int matmul_permuted(int n, const float *A, const float *B, float *C) {
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < n; k++) {
      // load B[j * n + k] into a register to avoid extra memory access
      // in inner loop
      float r = B[j * n + k];
      for (int i = 0; i < n; i++) {
        C[i + j * n] += A[i + k * n] * r;
      }
    }
  }

  return 0;
}

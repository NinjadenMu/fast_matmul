/**
 * @file matmul_naive.c
 * @brief a maximally naive matmul implementation, assumes column-major order
 *
 * This file assumes all buffers are in column-major order.  It's intended
 * to be a maximally naive baseline implementation.  As such, it should be
 * compiled with loop interchange and vectorization optimizations off.
 */

int matmul_naive(int n, const float *A, const float *B, float *C) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float acc = 0;
      for (int k = 0; k < n; k++) {
        acc += A[i + k * n] * B[j * n + k];
      }
      C[i + j * n] = acc;
    }
  }

  return 0;
}

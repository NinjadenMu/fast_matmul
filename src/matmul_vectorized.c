/**
 * @file matmul_vectorized.c
 *
 * This file assumes that all buffers are in column major order, and that C is
 * 0-initialized.  It defaults to a tile size of 64, which can be changed by
 * setting environment variable `tile_size`.
 *
 * Its microkernel operates on 128-bit registers using ARM NEON intrinisics.
 * The register block is 12x8, which uses 24 registers (and 3 additional
 * registers to load a column from A).  This can be changed by setting MR and
 * NR, although the default already uses registers very aggressively.  It
 * also uses tiling to improve cache utilization.  The tile size can be
 * modified at runtime by setting environment variable `tile_size`.
 * This implementation is already fairly fast, and can outperform OpenBLAS
 * when well-tuned in single-thread mode (although this is not a fair
 * comparision, since OpenBLAS may be highly parallelized.)
 */

#include <arm_neon.h>
#include <assert.h>
#include <stdlib.h>

#define UNROLL _Pragma("clang loop unroll(full)")

#ifndef MR
#define MR 12
#endif
#ifndef NR
#define NR 8
#endif

#ifndef MV
#define MV (MR / 4)
#endif

static inline void micro_kernel_vectorized(int n, int i, int j, int k_start,
                                           int k_end, float *restrict A,
                                           float *restrict B,
                                           float *restrict C) {
  float32x4_t acc[NR][MV];
  UNROLL
  for (int jj = 0; jj < NR; jj++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      acc[jj][vi] = vld1q_f32(&C[(j + jj) * n + i + vi * 4]);
    }
  }

  for (int k = k_start; k < k_end; k++) {
    float32x4_t vectorized_A_col[MV];
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      vectorized_A_col[vi] = vld1q_f32(&A[i + vi * 4 + k * n]);
    }

    UNROLL
    for (int jj = 0; jj < NR; jj++) {
      const float b = B[(j + jj) * n + k];
      UNROLL
      for (int vi = 0; vi < MV; vi++) {
        acc[jj][vi] = vfmaq_n_f32(acc[jj][vi], vectorized_A_col[vi], b);
      }
    }
  }

  UNROLL
  for (int jj = 0; jj < NR; jj++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      vst1q_f32(&C[i + vi * 4 + (j + jj) * n], acc[jj][vi]);
    }
  }
}

void matmul_vectorized(int n, float *restrict A, float *restrict B,
                       float *restrict C) {
  assert(!(MR % 4 || NR % 4));

  static int tile_size = 0;
  if (tile_size <= 0) {
    const char *env_tile_size_string = getenv("tile_size");
    tile_size = env_tile_size_string ? atoi(env_tile_size_string) : 96;
    assert(tile_size > 0);
  }

  for (int j_tile = 0; j_tile < n; j_tile += tile_size) {
    const int j_max = j_tile + tile_size < n ? j_tile + tile_size : n;
    for (int k_tile = 0; k_tile < n; k_tile += tile_size) {
      const int k_max = k_tile + tile_size < n ? k_tile + tile_size : n;
      for (int i_tile = 0; i_tile < n; i_tile += tile_size) {
        const int i_max = i_tile + tile_size < n ? i_tile + tile_size : n;
        for (int j = j_tile; j < j_max; j += NR) {
          for (int i = i_tile; i < i_max; i += MR) {
            if (i + MR <= i_max && j + NR <= j_max) {
              micro_kernel_vectorized(n, i, j, k_tile, k_max, A, B, C);
            } else {
              const int i_end = i + MR < i_max ? i + MR : i_max;
              const int j_end = j + NR < j_max ? j + NR : j_max;
              for (int jj = j; jj < j_end; jj++) {
                for (int k = k_tile; k < k_max; k++) {
                  for (int ii = i; ii < i_end; ii++) {
                    C[ii + jj * n] += A[ii + k * n] * B[jj * n + k];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
 * @file matmul_vectorized.c
 *
 * This file assumes that all buffers are in column major order, and that C is
 * 0-initialized.
 *
 * This implementation also uses tiling and register blocking. Furthermore,
 * its micro-kernel uses ARM NEON intrinsics to process data in 128-bit
 * (4 float) chunks, dramatically improving performance. Specifically,
 * columns of C and A are read 4 floats at a time along columns into
 * registers, and rank-1 updates are paralell over 4 floats from a column.
 *
 * The register block size is 12x8 by default, although the rows are read in
 * 4-float vectors, so only 24 registers are used. The block size may be
 * configured at compile-time by setting MR and NR, with MR divisible by 4.
 * The tile size may be configured at runtime by setting enviornment variable
 * `tile_size`, and should be divisible by both MR and NR (otherwise, the
 * edges of every tile will fall back to a slower matmul implementation.)
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
                                           int k_end, const float *restrict A,
                                           const float *restrict B,
                                           float *restrict C) {
  // This will be turned into scalar variables by the compiler,
  // which can be saved in registers.
  // They can be accessed by column and then by their chunk
  // within a column, where each chunk corresponds to a 4 float vector register.
  float32x4_t acc[NR][MV];
  UNROLL
  for (int jj = 0; jj < NR; jj++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      acc[jj][vi] = vld1q_f32(&C[(j + jj) * n + i + vi * 4]);
    }
  }

  // updates to the MRxNR block are computed as k rank-1 updates
  for (int k = k_start; k < k_end; k++) {
    // the column of A is read 4 floats at a time into vector registers
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
        // each value from B is broadcast into a 4 float vector to update
        // 4 rows at a time (since the A column is stored in 4-float chunks)
        acc[jj][vi] = vfmaq_n_f32(acc[jj][vi], vectorized_A_col[vi], b);
      }
    }
  }

  UNROLL
  for (int jj = 0; jj < NR; jj++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      // updated C values in registers are written back in 4-float chunks
      vst1q_f32(&C[i + vi * 4 + (j + jj) * n], acc[jj][vi]);
    }
  }
}

int matmul_vectorized(int n, const float *restrict A, const float *restrict B,
                      float *restrict C) {
  assert(!(MR % 4 || NR % 4));

  static int tile_size = 0;
  if (tile_size <= 0) {
    const char *env_tile_size_string = getenv("tile_size");
    tile_size = env_tile_size_string ? atoi(env_tile_size_string) : 96;
    assert(tile_size > 0);
  }

  // since essentially only the micro-kernel was changed, we continue using
  // the same 3 outer loops as the tiled implementation for cache efficiency
  // and the same 2 inner loops as the non-vectorized micro-kernel
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
            } 
            // if the tile is not fully coverable by MRxNR blocks, fallback to
            // the permuted matmul implementation for the remainder
            else {
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

  return 0;
}

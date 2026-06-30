/**
 * @file matmul_blis.c
 * @brief a fast single-threaded matmul following the BLIS formula
 *
 * This file assumes that all buffers are in column major order, and that C is
 * 0-initialized.
 *
 * Following the BLIS formula, this implementation allows users to configure
 * the panel (tile) sizes for A and B by setting mc, nc, and kc. This improves
 * on one uniform tile size by allowing panels to be sized to stay
 * cache-resident in different levels of the cache hierarchy.
 *
 * Furthermore, panels of A and B are packed when possible, which leads to
 * fewer TLB misses (since densely packed panels span fewer pages). Because
 * panels are packed in the order elements are accessed hardware prefetching
 * may also be improved, and it ensures that cache lines are fully utilized,
 * potentially enabling larger panel sizes.
 *
 * This implementation uses a vectorized, and register-blocked micro-kernel
 * performing `kc` rank-1 updates similar to the one in `matmul_vectorized.c`.
 *
 * Panel sizes may be configured at compile-time by setting environment
 * variables `mc` (should be divisible by MR),
 * `nc` (should be divisible by NR), and `kc`.
 * Sliver (micropanel/register block) sizes may be configured at compile time
 * by setting MR and NR. As before, MR should be divisible by 4.
 */

#include <arm_neon.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

#include <time.h>

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

static inline void micro_kernel_blis(int n, int i_start, int j_start,
                                     int k_start, int k_end,
                                     const float *restrict A_sliver,
                                     const float *restrict B_sliver,
                                     float *restrict C) {
  // This will be turned into scalar variables by the compiler,
  // which can be saved in registers.
  // They can be accessed by column and then by their chunk
  // within a column, where each chunk corresponds to a 4 float vector register.
  float32x4_t acc[NR][MV];
  UNROLL
  for (int j = 0; j < NR; j++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      acc[j][vi] = vld1q_f32(&C[i_start + vi * 4 + (j_start + j) * n]);
    }
  }

  const int kc = k_end - k_start;
  // updates to the MRxNR block are computed as `kc` rank-1 updates
  // note that `p` corresponds to the `k` index in the naive implementation
  for (int p = 0; p < kc; p++) {
    // computing the base address for slivers is easy because panels
    // are packed in the order elements are accessed by the micro-kernel
    const float *a_col = A_sliver + p * MR;
    const float *b_row = B_sliver + p * NR;

    float32x4_t vectorized_A_col[MV];
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      vectorized_A_col[vi] = vld1q_f32(a_col + vi * 4);
    }

    UNROLL
    for (int j = 0; j < NR; j++) {
      const float b = b_row[j];
      UNROLL
      for (int vi = 0; vi < MV; vi++) {
        // each value from B is broadcast into a 4 float vector to update
        // 4 rows at a time (since the A column is stored in 4-float chunks)
        acc[j][vi] = vfmaq_n_f32(acc[j][vi], vectorized_A_col[vi], b);
      }
    }
  }

  UNROLL
  for (int j = 0; j < NR; j++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      // updated C values in registers are written back in 4-float chunks
      vst1q_f32(&C[i_start + vi * 4 + (j_start + j) * n], acc[j][vi]);
    }
  }
}

/*
 * pack_A is friendly for vectorization since A is read along columns and
 * is stored in column major order.
 *
 * Note that the loop nest and indexing reflect the order elements from A are
 * accessed by the micro-kernel: first by sliver, then by column, then by row.
 */
static void pack_A(int n, int i_start, int i_end, int k_start, int k_end,
                   const float *restrict A, float *restrict A_pack) {
  float *restrict dst = A_pack;
  for (int i = i_start; i + MR <= i_end; i += MR) {
    for (int p = k_start; p < k_end; p++) {
      const float *active_col_ptr = A + i + p * n;
      UNROLL
      for (int vi = 0; vi < MV; vi++) {
        vst1q_f32(dst + vi * 4, vld1q_f32(active_col_ptr + vi * 4));
      }

      dst += MR;
    }
  }
}

/*
 * pack_B is not vectorized because elements from B should be packed by
 * moving along columns, which is a strided access pattern.
 * While vectorization is possible, it requires some complexity which
 * I think isn't worth it given that packing work is heavily amortized.
 *
 * Note that the loop nest and indexing reflect the order elements from B are
 * accessed by the micro-kernel: first by sliver, then by row, then by column.
 */
static void pack_B(int n, int j_start, int j_end, int k_start, int k_end,
                   const float *restrict B, float *restrict B_pack) {
  float *restrict dst = B_pack;
  for (int j = j_start; j + NR <= j_end; j += NR) {
    for (int p = k_start; p < k_end; p++) {
      for (int jj = 0; jj < NR; jj++) {
        *dst++ = B[(j + jj) * n + p];
      }
    }
  }
}

int matmul_blis(int n, const float *restrict A, const float *restrict B,
                float *restrict C) {
  static bool initialized = false;
  static int mc = 0;
  static int nc = 0;
  static int kc = 0;
  if (!initialized) {
    mc = get_env_int("mc", 3300);
    nc = get_env_int("nc", 4096);
    kc = get_env_int("kc", 2048);
    assert(!(mc % MR));
    assert(!(nc % NR));
    initialized = true;
  }

  float *A_pack;
  float *B_pack;
  if (posix_memalign((void **)&A_pack, MEM_ALIGNMENT_MIN,
                     mc * kc * sizeof(float))) {
    fprintf(stderr, "Error: memory allocation failed");
    return 1;
  } else if (posix_memalign((void **)&B_pack, MEM_ALIGNMENT_MIN,
                            nc * kc * sizeof(float))) {
    fprintf(stderr, "Error: memory allocation failed");
    return 1;
  }

  // the outer 3 loops slice A and B into panels, which also determine
  // the way C is tiled
  for (int jc = 0; jc < n; jc += nc) {
    const int j_max = jc + nc < n ? jc + nc : n;
    for (int pc = 0; pc < n; pc += kc) {
      const int k_max = pc + kc < n ? pc + kc : n;
      // B is packed as soon as its panel is determined
      // the cost of packing is amortized across the inner loops
      pack_B(n, jc, j_max, pc, k_max, B, B_pack);

      for (int ic = 0; ic < n; ic += mc) {
        const int i_max = ic + mc < n ? ic + mc : n;
        // B is packed as soon as its panel is determined
        // the cost of packing is amortized across the inner loops
        pack_A(n, ic, i_max, pc, k_max, A, A_pack);

        const float *B_sliver = B_pack;
        for (int jr = jc; jr < j_max; jr += NR) {
          const float *A_sliver = A_pack;
          for (int ir = ic; ir < i_max; ir += MR) {
            if (ir + MR <= i_max && jr + NR <= j_max) {
              micro_kernel_blis(n, ir, jr, pc, k_max, A_sliver, B_sliver, C);
            } 
            // if the tile is not fully coverable by MRxNR blocks, fallback to
            // the permuted matmul implementation for the remainder
            else {
              const int i_end = ir + MR < i_max ? ir + MR : i_max;
              const int j_end = jr + NR < j_max ? jr + NR : j_max;
              for (int j = jr; j < j_end; j++) {
                for (int p = pc; p < k_max; p++) {
                  for (int i = ir; i < i_end; i++) {
                    C[i + j * n] += A[i + p * n] * B[j * n + p];
                  }
                }
              }
            }
            A_sliver += MR * (k_max - pc);
          }
          B_sliver += NR * (k_max - pc);
        }
      }
    }
  }

  free(A_pack);
  free(B_pack);
  return 0;
}

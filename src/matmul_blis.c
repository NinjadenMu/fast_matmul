#include <arm_neon.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

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
  float32x4_t acc[NR][MV];
  UNROLL
  for (int j = 0; j < NR; j++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      acc[j][vi] = vld1q_f32(&C[i_start + vi * 4 + (j_start + j) * n]);
    }
  }

  const int kc = k_end - k_start;
  for (int p = 0; p < kc; p++) {
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
        acc[j][vi] = vfmaq_n_f32(acc[j][vi], vectorized_A_col[vi], b);
      }
    }
  }

  UNROLL
  for (int j = 0; j < NR; j++) {
    UNROLL
    for (int vi = 0; vi < MV; vi++) {
      vst1q_f32(&C[i_start + vi * 4 + (j_start + j) * n], acc[j][vi]);
    }
  }
}

static int get_env_int(const char *env_var, int default_val) {
  const char *val_string = getenv(env_var);
  int val = val_string ? atoi(val_string) : default_val;
  assert(val > 0);
  return val;
}

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
    mc = get_env_int("mc", 96);
    nc = get_env_int("nc", 96);
    kc = get_env_int("kc", 96);
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

  for (int jc = 0; jc < n; jc += nc) {
    const int j_max = jc + nc < n ? jc + nc : n;
    for (int pc = 0; pc < n; pc += kc) {
      const int k_max = pc + kc < n ? pc + kc : n;
      pack_B(n, jc, j_max, pc, k_max, B, B_pack);

      for (int ic = 0; ic < n; ic += mc) {
        const int i_max = ic + mc < n ? ic + mc : n;
        pack_A(n, ic, i_max, pc, k_max, A, A_pack);

        const float *B_sliver = B_pack;
        for (int jr = jc; jr < j_max; jr += NR) {
          const float *A_sliver = A_pack;
          for (int ir = ic; ir < i_max; ir += MR) {
            if (ir + MR <= i_max && jr + NR <= j_max) {
              micro_kernel_blis(n, ir, jr, pc, k_max, A_sliver, B_sliver, C);
            } else {
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

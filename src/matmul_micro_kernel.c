/**
 * @file matmul_micro_kernel.c
 * @brief A fast matmul implementation using tiling and register blocking
 *
 * This file assumes that all buffers are in column major order, and that C is
 * 0-initialized.  It defaults to a tile size of 64, which can be changed by
 * setting environment variable `tile_size`.  The register block size is 4x4,
 * which can be changed by setting compile-time macros `MR` and `NR`.  It uses
 * tiling to improve cache utilization, as cache resident data is used
 * maximally.  Then, because a naive implementation is bound by LD/ST
 * pipes, this implementation uses a micro-kernel that loads small blocks
 * fully into registers, which reduces traffic between the CPU and the cache.
 * Previously, a single FMA (fused-multiply-add) required 2 loads and a store,
 * which is now heavily amortized.  The micro-kernel also does matmul
 * following a k-i-j traversal order instead of j-k-i, since register reuse
 * and ILP are the biggest bottlenecks, not cache efficiency.
 *
 * Although we now have a relatively fast implementation, autovectorization
 * should still be turned off when compiling.
 */

#include <assert.h>
#include <stdlib.h>

#define UNROLL _Pragma("clang loop unroll(full)")

#ifndef MR
#define MR 4
#endif
#ifndef NR
#define NR 4
#endif

/**
 * @brief Fully computes a MR x NR block
 *
 * By loading a block of C into registers, and accumulating into
 * registers (only writing back once in the end), LD/ST ops are
 * heavily amortized, removing a significant bottleneck.
 *
 * Within a block, values are computed using rank-1 updates (taking a column
 * from A and a row from B), which allows values from A and B to be stored in
 * registers as well (since this means MR/KR values are loaded from them,
 * not tile_size).
 */
static inline void micro_kernel(int n, int i, int j, int k_start, int k_end,
                                const float *restrict A,
                                const float *restrict B, float *restrict C) {

  // This will be turned into scalar variables by the compiler,
  // which can be saved in registers
  // We can write to these instead of C, saving a LD/ST instruction
  float acc[MR][NR];
  // Load block from C into registers
  UNROLL
  for (int ii = 0; ii < MR; ii++) {
    UNROLL
    for (int jj = 0; jj < NR; jj++) {
      acc[ii][jj] = C[(i + ii) + (j + jj) * n];
    }
  }

  // By iterating k-i-j, we compute the matrix as k rank-1 updates
  // Each outer loop takes MR loads from A and NR loads from B,
  // which are amortized by MR x NR FMAs.
  // If instead we used j-k-i, each column of B would be amortized by MR,
  // but rows of A would have to be re-loaded NR times.
  // Furthermore, k-i-j gives us MR x NR dependency chains (instead of NR),
  // which improves latency hiding
  for (int k = k_start; k < k_end; k++) {
    // Used to store k-th column and row from A and B respectively in registers
    //float A_col[MR];
    float B_row[NR];
    /*UNROLL
    for (int ii = 0; ii < MR; ii++) {
      A_col[ii] = A[i + ii + k * n];
    }*/

    UNROLL
    for (int jj = 0; jj < NR; jj++) {
      B_row[jj] = B[(j + jj) * n + k];
    }

    UNROLL
    for (int ii = 0; ii < MR; ii++) {
      float a = A[i + ii + k * n];
      UNROLL
      for (int jj = 0; jj < NR; jj++) {
        //acc[ii][jj] += A_col[ii] * B_row[jj];
        acc[ii][jj] += a * B_row[jj];
      }
    }
  }

  UNROLL
  for (int ii = 0; ii < MR; ii++) {
    UNROLL
    for (int jj = 0; jj < NR; jj++) {
      C[i + ii + (j + jj) * n] = acc[ii][jj];
    }
  }
}

void matmul_micro_kernel(int n, const float *restrict A,
                         const float *restrict B, float *restrict C) {
  static int tile_size = 0;

  // avoids re-reading the tile size every time this function is called
  if (tile_size <= 0) {
    const char *env_tile_size_string = getenv("tile_size");
    tile_size = env_tile_size_string ? atoi(env_tile_size_string) : 64;
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
              micro_kernel(n, i, j, k_tile, k_max, A, B, C);
            } 
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
}

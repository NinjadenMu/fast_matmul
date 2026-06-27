/**
 * @file matmul_tiled.c
 * @brief a matmul implementation with tiling, has good cache utilization
 *
 * This file assumes all buffers are in column-major order, and that the
 * output buffer is 0-initialized.  It defaults to a tile size of 64, which
 * can be changed by setting environment variable `tile_size`.  Well-chosen
 * tile sizes can lead to much better cache utilization by keeping all
 * floats in the cache until they're not needed anymore.
 * However, this is actually slower than the permuted version on my hardware,
 * since both are bottlenecked by the number of LD/ST pipes, not the cache.
 * This is still designed to be a simple implementation, so autovectorization
 * should be turned off when compiling.
 */

#include <assert.h>
#include <stdlib.h>

int matmul_tiled(int n, const float *A, const float *B, float *C) {
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
        for (int j = j_tile; j < j_max; j++) {
          for (int k = k_tile; k < k_max; k++) {
            // load B[j * n + k] into a register to avoid extra memory access
            // in inner loop
            float r = B[j * n + k];
            for (int i = i_tile; i < i_max; i++) {
              C[i + j * n] += A[i + k * n] * r;
            }
          }
        }
      }
    }
  }

  return 0;
}

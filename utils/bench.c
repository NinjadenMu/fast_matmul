#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matmul.h"

typedef struct {
  const char *name;
  matmul_func_t func;
} impl_t;

impl_t implementations[] = {
    {"naive", matmul_naive},           {"permuted", matmul_permuted},
    {"tiled", matmul_tiled},           {"micro_kernel", matmul_micro_kernel},
    {"vectorized", matmul_vectorized}, {"blis", matmul_blis},
    {"parallel", matmul_parallel},     {NULL, NULL}};

void randomize_matrix(int n, float *matrix) {
  for (int i = 0; i < n * n; i++) {
    matrix[i] = (float)rand() / RAND_MAX;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "Error: not all required arguments provided");
    return 1;
  }

  const char *impl_name = argv[1];
  int n = atoi(argv[2]);
  int num_trials = atoi(argv[3]);

  matmul_func_t func = NULL;
  for (int i = 0; implementations[i].name != NULL; i++) {
    if (strcmp(implementations[i].name, impl_name) == 0) {
      func = implementations[i].func;
      break;
    }
  }

  if (!func) {
    fprintf(stderr, "Error: unknown implementation `%s`\n", impl_name);
    return 1;
  }

  float *A, *B, *C;
  size_t matrix_bytes = n * n * sizeof(float);
  // allocate 64-byte aligned memory
  if (posix_memalign((void **)&A, MEM_ALIGNMENT_MIN, matrix_bytes) ||
      posix_memalign((void **)&B, MEM_ALIGNMENT_MIN, matrix_bytes) ||
      posix_memalign((void **)&C, MEM_ALIGNMENT_MIN, matrix_bytes)) {
    fprintf(stderr, "Error: memory allocation failed\n");
    return 1;
  }
  randomize_matrix(n, A);
  randomize_matrix(n, B);
  memset(C, 0, matrix_bytes);

  double total_time = 0;
  for (int i = 0; i < num_trials + 1; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    func(n, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

    if (i != 0)
      total_time += elapsed;
  }

  double gflops = (2.0 * n * n * n * num_trials) / (1e9 * total_time);
  printf("{\"avg_time\":\"%.3fs\",\"gflops\":\"%.3f\"}",
         total_time / (num_trials), gflops);
  fflush(stdout);

  free(A);
  free(B);
  free(C);

  return 0;
}

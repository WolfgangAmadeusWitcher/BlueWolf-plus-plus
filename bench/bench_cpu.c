#include "bwpp_cpu_ref.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_matrix(float *dst, uint32_t rows, uint32_t cols, float scale) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      dst[i * cols + j] = (float)(i * cols + j + 1) * scale;
    }
  }
}

static void parse_meta(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "bench: failed to read %s\n", path);
    return;
  }
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "bwpp.meta:") || strstr(line, "bwpp.plan:")) {
      fputs(line, stdout);
    }
  }
  fclose(f);
}

int main(int argc, char **argv) {
  uint32_t M = 256, N = 256, K = 256;
  uint32_t iters = 10;
  const char *metal_path = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      M = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      N = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      K = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--metal") == 0 && i + 1 < argc) {
      metal_path = argv[++i];
    }
  }

  if (metal_path) {
    printf("== MSL metadata ==\n");
    parse_meta(metal_path);
  }

  float *a = (float *)malloc(sizeof(float) * M * K);
  float *b = (float *)malloc(sizeof(float) * K * N);
  float *c = (float *)malloc(sizeof(float) * M * N);
  float *bias = (float *)malloc(sizeof(float) * N);
  if (!a || !b || !c || !bias) {
    fprintf(stderr, "bench: alloc failed\n");
    free(a);
    free(b);
    free(c);
    free(bias);
    return 1;
  }
  fill_matrix(a, M, K, 0.01f);
  fill_matrix(b, K, N, 0.02f);
  for (uint32_t i = 0; i < M * N; ++i) {
    c[i] = 0.0f;
  }
  for (uint32_t i = 0; i < N; ++i) {
    bias[i] = 0.0f;
  }

  double t0 = now_sec();
  for (uint32_t i = 0; i < iters; ++i) {
    bwpp_cpu_matmul_f32(a, b, c, M, N, K, K, N, N, bias, 0, 0);
  }
  double t1 = now_sec();
  double secs = t1 - t0;
  double flops = 2.0 * (double)M * (double)N * (double)K * (double)iters;
  printf("matmul: M=%u N=%u K=%u iters=%u time=%.6fs gflops=%.2f\n",
         M, N, K, iters, secs, (flops / 1e9) / secs);

  uint32_t rows = 256;
  uint32_t cols = 256;
  float *x = (float *)malloc(sizeof(float) * rows * cols);
  float *y = (float *)malloc(sizeof(float) * rows * cols);
  float *z = (float *)malloc(sizeof(float) * rows * cols);
  float *gamma = (float *)malloc(sizeof(float) * cols);
  if (!x || !y || !z || !gamma) {
    fprintf(stderr, "bench: alloc failed for norm buffers\n");
    free(a);
    free(b);
    free(c);
    free(bias);
    free(x);
    free(y);
    free(z);
    free(gamma);
    return 1;
  }
  fill_matrix(x, rows, cols, 0.01f);
  for (uint32_t i = 0; i < rows * cols; ++i) {
    y[i] = 0.0f;
    z[i] = 0.0f;
  }
  for (uint32_t i = 0; i < cols; ++i) {
    gamma[i] = 1.0f;
  }

  t0 = now_sec();
  for (uint32_t i = 0; i < iters; ++i) {
    bwpp_cpu_softmax_f32(x, y, rows, cols, cols);
  }
  t1 = now_sec();
  printf("softmax: rows=%u cols=%u iters=%u time=%.6fs\n", rows, cols, iters, t1 - t0);

  t0 = now_sec();
  for (uint32_t i = 0; i < iters; ++i) {
    bwpp_cpu_rmsnorm_f32(x, z, gamma, NULL, rows, cols, cols, 1e-5f);
  }
  t1 = now_sec();
  printf("rmsnorm: rows=%u cols=%u iters=%u time=%.6fs\n", rows, cols, iters, t1 - t0);

  free(a);
  free(b);
  free(c);
  free(bias);
  free(x);
  free(y);
  free(z);
  free(gamma);
  return 0;
}

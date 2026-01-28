#include "bwpp_cpu_ref.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_matrix(float *dst, uint32_t rows, uint32_t cols, float scale) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      dst[i * cols + j] = (float)(i * cols + j + 1) * scale;
    }
  }
}

int main(void) {
  const uint32_t rows = 2;
  const uint32_t cols = 4;
  float x[rows * cols];
  float y[rows * cols];
  float z[rows * cols];
  float gamma[cols];

  fill_matrix(x, rows, cols, 0.1f);
  for (uint32_t i = 0; i < rows * cols; ++i) {
    y[i] = 0.0f;
    z[i] = 0.0f;
  }
  for (uint32_t i = 0; i < cols; ++i) {
    gamma[i] = 1.0f;
  }

  bwpp_cpu_softmax_f32(x, y, rows, cols, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    float sum = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      sum += y[r * cols + c];
    }
    if (fabsf(sum - 1.0f) > 1e-3f) {
      fprintf(stderr, "softmax row sum mismatch: %.6f\n", sum);
      return 1;
    }
  }

  bwpp_cpu_rmsnorm_f32(x, z, gamma, rows, cols, cols, 1e-5f);
  for (uint32_t r = 0; r < rows; ++r) {
    float sumsq = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      float v = z[r * cols + c];
      sumsq += v * v;
    }
    float rms = sqrtf(sumsq / (float)cols);
    if (fabsf(rms - 1.0f) > 1e-2f) {
      fprintf(stderr, "rmsnorm rms mismatch: %.6f\n", rms);
      return 1;
    }
  }

  printf("CPU PASS softmax+rmsnorm\n");
  return 0;
}

#include "bwpp_cpu_ref.h"
#include <math.h>
#include <stdio.h>

static void fill_matrix(float *x, uint32_t rows, uint32_t cols) {
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      x[r * cols + c] = (float)(r * cols + c);
    }
  }
}

static int check_mask_axis1(void) {
  const uint32_t rows = 2;
  const uint32_t cols = 3;
  float x[rows * cols];
  float mask[rows * cols];
  fill_matrix(x, rows, cols);
  bwpp_cpu_reduce_max_mask_f32(x, mask, rows, cols, 1);
  int ok = 1;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      float expected = (c == cols - 1) ? 1.0f : 0.0f;
      if (fabsf(mask[r * cols + c] - expected) > 1e-6f) {
        ok = 0;
      }
    }
  }
  return ok;
}

static int check_grad_axis1(void) {
  const uint32_t rows = 2;
  const uint32_t cols = 3;
  float x[rows * cols];
  float mask[rows * cols];
  float dy[rows];
  float dx[rows * cols];
  fill_matrix(x, rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    dy[r] = (float)(r + 1);
  }
  bwpp_cpu_reduce_max_mask_f32(x, mask, rows, cols, 1);
  bwpp_cpu_reduce_max_grad_f32(mask, dy, dx, rows, cols, 1);
  int ok = 1;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      float expected = (c == cols - 1) ? dy[r] : 0.0f;
      if (fabsf(dx[r * cols + c] - expected) > 1e-6f) {
        ok = 0;
      }
    }
  }
  return ok;
}

int main(void) {
  if (!check_mask_axis1()) {
    fprintf(stderr, "FAIL reduce_max mask axis=1\n");
    return 1;
  }
  if (!check_grad_axis1()) {
    fprintf(stderr, "FAIL reduce_max grad axis=1\n");
    return 1;
  }
  printf("CPU PASS reduce_max mask+grad\n");
  return 0;
}

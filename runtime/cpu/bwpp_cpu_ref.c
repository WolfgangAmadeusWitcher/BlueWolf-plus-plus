#include "bwpp_cpu_ref.h"
#include <math.h>

static float bwpp_silu(float x) {
  return x / (1.0f + expf(-x));
}

void bwpp_cpu_matmul_f32(const float *a,
                         const float *b,
                         float *c,
                         uint32_t M,
                         uint32_t N,
                         uint32_t K,
                         uint32_t lda,
                         uint32_t ldb,
                         uint32_t ldc,
                         const float *bias,
                         int apply_silu,
                         int apply_bias) {
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        acc += a[row * lda + k] * b[k * ldb + col];
      }
      if (apply_bias && bias) {
        acc += bias[col];
      }
      if (apply_silu) {
        acc = bwpp_silu(acc);
      }
      c[row * ldc + col] = acc;
    }
  }
}

void bwpp_cpu_softmax_f32(const float *x,
                          float *y,
                          uint32_t rows,
                          uint32_t cols,
                          uint32_t ld) {
  for (uint32_t r = 0; r < rows; ++r) {
    float maxv = -INFINITY;
    for (uint32_t c = 0; c < cols; ++c) {
      float v = x[r * ld + c];
      if (v > maxv) {
        maxv = v;
      }
    }
    float sum = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      float e = expf(x[r * ld + c] - maxv);
      y[r * ld + c] = e;
      sum += e;
    }
    float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      y[r * ld + c] *= inv;
    }
  }
}

void bwpp_cpu_rmsnorm_f32(const float *x,
                          float *y,
                          const float *gamma,
                          const float *beta,
                          uint32_t rows,
                          uint32_t cols,
                          uint32_t ld,
                          float eps) {
  for (uint32_t r = 0; r < rows; ++r) {
    float sumsq = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      float v = x[r * ld + c];
      sumsq += v * v;
    }
    float mean = sumsq / (float)cols;
    float inv = 1.0f / sqrtf(mean + eps);
    for (uint32_t c = 0; c < cols; ++c) {
      float v = x[r * ld + c] * inv;
      if (gamma) {
        v *= gamma[c];
      }
      if (beta) {
        v += beta[c];
      }
      y[r * ld + c] = v;
    }
  }
}

void bwpp_cpu_reduce_max_mask_f32(const float *x,
                                  float *mask,
                                  uint32_t rows,
                                  uint32_t cols,
                                  int axis) {
  if (!x || !mask) {
    return;
  }
  for (uint32_t i = 0; i < rows * cols; ++i) {
    mask[i] = 0.0f;
  }
  if (axis == 0) {
    for (uint32_t c = 0; c < cols; ++c) {
      float maxv = -INFINITY;
      for (uint32_t r = 0; r < rows; ++r) {
        float v = x[r * cols + c];
        if (v > maxv) {
          maxv = v;
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        float v = x[r * cols + c];
        if (v == maxv) {
          mask[r * cols + c] = 1.0f;
        }
      }
    }
  } else {
    for (uint32_t r = 0; r < rows; ++r) {
      float maxv = -INFINITY;
      for (uint32_t c = 0; c < cols; ++c) {
        float v = x[r * cols + c];
        if (v > maxv) {
          maxv = v;
        }
      }
      for (uint32_t c = 0; c < cols; ++c) {
        float v = x[r * cols + c];
        if (v == maxv) {
          mask[r * cols + c] = 1.0f;
        }
      }
    }
  }
}

void bwpp_cpu_reduce_max_grad_f32(const float *mask,
                                  const float *dy,
                                  float *dx,
                                  uint32_t rows,
                                  uint32_t cols,
                                  int axis) {
  if (!mask || !dy || !dx) {
    return;
  }
  for (uint32_t i = 0; i < rows * cols; ++i) {
    dx[i] = 0.0f;
  }
  if (axis == 0) {
    for (uint32_t c = 0; c < cols; ++c) {
      float g = dy[c];
      for (uint32_t r = 0; r < rows; ++r) {
        dx[r * cols + c] = mask[r * cols + c] * g;
      }
    }
  } else {
    for (uint32_t r = 0; r < rows; ++r) {
      float g = dy[r];
      for (uint32_t c = 0; c < cols; ++c) {
        dx[r * cols + c] = mask[r * cols + c] * g;
      }
    }
  }
}

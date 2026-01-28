#include "bwpp_cpu_ref.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long size = ftell(f);
  if (size < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  char *buf = (char *)malloc((size_t)size + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t read = fread(buf, 1, (size_t)size, f);
  fclose(f);
  buf[read] = '\0';
  if (out_len) {
    *out_len = read;
  }
  return buf;
}

static void fill_matrix(float *dst, uint32_t rows, uint32_t cols, float scale) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      dst[i * cols + j] = (float)(i * cols + j + 1) * scale;
    }
  }
}

static float bwpp_silu(float x) {
  return x / (1.0f + expf(-x));
}

static void parse_epilogue(const char *src, int *ep_add, int *ep_silu) {
  *ep_add = 0;
  *ep_silu = 0;
  if (!src) {
    return;
  }
  if (strstr(src, "#define BWPP_EPILOGUE_ADD 1")) {
    *ep_add = 1;
  }
  if (strstr(src, "#define BWPP_EPILOGUE_SILU 1")) {
    *ep_silu = 1;
  }
  if (*ep_add || *ep_silu) {
    return;
  }
  const char *p = strstr(src, "bwpp.meta: epilogue=");
  if (!p) {
    return;
  }
  p += strlen("bwpp.meta: epilogue=");
  if (strncmp(p, "add_silu", 8) == 0) {
    *ep_add = 1;
    *ep_silu = 1;
  } else if (strncmp(p, "add", 3) == 0) {
    *ep_add = 1;
  } else if (strncmp(p, "silu", 4) == 0) {
    *ep_silu = 1;
  }
}

static int test_matmul(const char *src) {
  if (!strstr(src, "bwpp.meta: kernel=matmul_f16")) {
    return 0;
  }
  int ep_add = 0;
  int ep_silu = 0;
  parse_epilogue(src, &ep_add, &ep_silu);

  const uint32_t M = 4;
  const uint32_t N = 4;
  const uint32_t K = 4;
  float a[M * K];
  float b[K * N];
  float c[M * N];
  float bias[N];
  float ref[M * N];
  fill_matrix(a, M, K, 0.1f);
  fill_matrix(b, K, N, 0.05f);
  for (uint32_t i = 0; i < M * N; ++i) {
    c[i] = 0.0f;
    ref[i] = 0.0f;
  }
  for (uint32_t i = 0; i < N; ++i) {
    bias[i] = ep_add ? (0.01f * (float)(i + 1)) : 0.0f;
  }
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        acc += a[row * K + k] * b[k * N + col];
      }
      float out = acc;
      if (ep_add) {
        out += bias[col];
      }
      if (ep_silu) {
        out = bwpp_silu(out);
      }
      ref[row * N + col] = out;
    }
  }
  bwpp_cpu_matmul_f32(a, b, c, M, N, K, K, N, N, bias, ep_silu, ep_add);
  float max_err = 0.0f;
  for (uint32_t i = 0; i < M * N; ++i) {
    float diff = fabsf(c[i] - ref[i]);
    if (diff > max_err) {
      max_err = diff;
    }
  }
  if (max_err > 1e-4f) {
    fprintf(stderr, "CPU FAIL matmul max_err=%.6f ep_add=%d ep_silu=%d\n",
            max_err, ep_add, ep_silu);
    return -1;
  }
  printf("CPU PASS matmul max_err=%.6f ep_add=%d ep_silu=%d\n", max_err, ep_add, ep_silu);
  return 1;
}

static int test_softmax(const char *src) {
  if (!strstr(src, "bwpp.meta: aux_kernel=softmax_f16")) {
    return 0;
  }
  const uint32_t rows = 2;
  const uint32_t cols = 4;
  const uint32_t ld = cols;
  float x[rows * cols];
  float y[rows * cols];
  float ref[rows * cols];
  fill_matrix(x, rows, cols, 0.1f);
  for (uint32_t i = 0; i < rows * cols; ++i) {
    y[i] = 0.0f;
    ref[i] = 0.0f;
  }
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
      ref[r * ld + c] = e;
      sum += e;
    }
    float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      ref[r * ld + c] *= inv;
    }
  }
  bwpp_cpu_softmax_f32(x, y, rows, cols, ld);
  float max_err = 0.0f;
  for (uint32_t i = 0; i < rows * cols; ++i) {
    float diff = fabsf(y[i] - ref[i]);
    if (diff > max_err) {
      max_err = diff;
    }
  }
  if (max_err > 1e-5f) {
    fprintf(stderr, "CPU FAIL softmax max_err=%.6f\n", max_err);
    return -1;
  }
  printf("CPU PASS softmax max_err=%.6f\n", max_err);
  return 1;
}

static int test_rmsnorm(const char *src) {
  if (!strstr(src, "bwpp.meta: aux_kernel=rmsnorm_f16")) {
    return 0;
  }
  const uint32_t rows = 2;
  const uint32_t cols = 4;
  const uint32_t ld = cols;
  float x[rows * cols];
  float y[rows * cols];
  float ref[rows * cols];
  float gamma[cols];
  fill_matrix(x, rows, cols, 0.1f);
  for (uint32_t i = 0; i < rows * cols; ++i) {
    y[i] = 0.0f;
    ref[i] = 0.0f;
  }
  for (uint32_t i = 0; i < cols; ++i) {
    gamma[i] = 1.0f;
  }
  const float eps = 1e-5f;
  for (uint32_t r = 0; r < rows; ++r) {
    float sumsq = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      float v = x[r * ld + c];
      sumsq += v * v;
    }
    float inv = 1.0f / sqrtf(sumsq / (float)cols + eps);
    for (uint32_t c = 0; c < cols; ++c) {
      ref[r * ld + c] = x[r * ld + c] * inv * gamma[c];
    }
  }
  bwpp_cpu_rmsnorm_f32(x, y, gamma, rows, cols, ld, eps);
  float max_err = 0.0f;
  for (uint32_t i = 0; i < rows * cols; ++i) {
    float diff = fabsf(y[i] - ref[i]);
    if (diff > max_err) {
      max_err = diff;
    }
  }
  if (max_err > 1e-5f) {
    fprintf(stderr, "CPU FAIL rmsnorm max_err=%.6f\n", max_err);
    return -1;
  }
  printf("CPU PASS rmsnorm max_err=%.6f\n", max_err);
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output.metal>\n", argv[0]);
    return 1;
  }
  size_t len = 0;
  char *src = read_file(argv[1], &len);
  if (!src) {
    fprintf(stderr, "failed to read: %s\n", argv[1]);
    return 1;
  }
  int ran = 0;
  int rc = 0;
  int r = test_matmul(src);
  if (r != 0) {
    ran = 1;
    if (r < 0) {
      rc = 1;
    }
  }
  r = test_softmax(src);
  if (r != 0) {
    ran = 1;
    if (r < 0) {
      rc = 1;
    }
  }
  r = test_rmsnorm(src);
  if (r != 0) {
    ran = 1;
    if (r < 0) {
      rc = 1;
    }
  }
  free(src);
  if (!ran) {
    fprintf(stderr, "no kernels found in %s\n", argv[1]);
    return 1;
  }
  return rc;
}

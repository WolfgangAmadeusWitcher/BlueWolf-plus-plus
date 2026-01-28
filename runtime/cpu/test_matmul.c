#include "bwpp_cpu_ref.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_matrix(float *dst, uint32_t rows, uint32_t cols, float scale) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      dst[i * cols + j] = (float)(i * cols + j + 1) * scale;
    }
  }
}

static void detect_epilogue(const char *src, int *ep_add, int *ep_silu) {
  *ep_add = 0;
  *ep_silu = 0;
  if (!src) {
    return;
  }
  if (strstr(src, "add(") && strstr(src, "bias")) {
    *ep_add = 1;
  }
  if (strstr(src, "silu")) {
    *ep_silu = 1;
  }
}

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

int main(int argc, char **argv) {
  const char *path = "examples/matmul.bwpp";
  if (argc > 1) {
    path = argv[1];
  }

  size_t len = 0;
  char *src = read_file(path, &len);
  if (!src) {
    fprintf(stderr, "failed to read: %s\n", path);
    return 1;
  }

  int ep_add = 0;
  int ep_silu = 0;
  detect_epilogue(src, &ep_add, &ep_silu);
  free(src);

  const uint32_t M = 4;
  const uint32_t N = 4;
  const uint32_t K = 4;

  float a[M * K];
  float b[K * N];
  float c[M * N];
  float bias[N];

  fill_matrix(a, M, K, 0.1f);
  fill_matrix(b, K, N, 0.05f);
  for (uint32_t i = 0; i < M * N; ++i) {
    c[i] = 0.0f;
  }
  for (uint32_t i = 0; i < N; ++i) {
    bias[i] = ep_add ? (0.01f * (float)(i + 1)) : 0.0f;
  }

  bwpp_cpu_matmul_f32(a, b, c, M, N, K, K, N, N, bias, ep_silu, ep_add);

  float checksum = 0.0f;
  for (uint32_t i = 0; i < M * N; ++i) {
    checksum += c[i];
  }

  printf("CPU PASS checksum=%.6f ep_add=%d ep_silu=%d\n", checksum, ep_add, ep_silu);
  return 0;
}

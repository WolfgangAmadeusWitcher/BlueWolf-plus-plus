#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#import "dispatch_stub.h"

typedef __fp16 half;

static double now_sec(void) {
  static mach_timebase_info_data_t info;
  if (info.denom == 0) {
    mach_timebase_info(&info);
  }
  uint64_t t = mach_absolute_time();
  double ns = (double)t * (double)info.numer / (double)info.denom;
  return ns * 1e-9;
}

static void fill_matrix(half *dst, uint32_t rows, uint32_t cols, float scale) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      float v = (float)(i * cols + j + 1) * scale;
      dst[i * cols + j] = (half)v;
    }
  }
}

static int has_kernel(const char *src, const char *needle) {
  if (!src || !needle) {
    return 0;
  }
  return strstr(src, needle) != NULL;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    NSString *mslPath = @"out_attention.metal";
    uint32_t iters = 10;
    uint32_t M = 256, N = 256, K = 256, D = 256;
    uint32_t rows = 256, cols = 256;
    int skip_matmul = 0;
    int skip_softmax = 0;
    int skip_rmsnorm = 0;
    int skip_attention = 0;

    for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "--msl") == 0 && i + 1 < argc) {
        mslPath = [NSString stringWithUTF8String:argv[++i]];
      } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
        iters = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
        M = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
        N = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        K = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
        D = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
        rows = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
        cols = (uint32_t)strtoul(argv[++i], NULL, 10);
      } else if (strcmp(argv[i], "--skip-matmul") == 0) {
        skip_matmul = 1;
      } else if (strcmp(argv[i], "--skip-softmax") == 0) {
        skip_softmax = 1;
      } else if (strcmp(argv[i], "--skip-rmsnorm") == 0) {
        skip_rmsnorm = 1;
      } else if (strcmp(argv[i], "--skip-attention") == 0) {
        skip_attention = 1;
      }
    }

    NSError *err = nil;
    NSString *mslSource = [NSString stringWithContentsOfFile:mslPath
                                                   encoding:NSUTF8StringEncoding
                                                      error:&err];
    if (!mslSource) {
      fprintf(stderr, "failed to read MSL: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    const char *mslC = [mslSource UTF8String];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    if (!skip_matmul && has_kernel(mslC, "bwpp.meta: kernel=matmul_f16")) {
      BwppMatmulParams params = { M, N, K, K, N, N };
      id<MTLBuffer> a = [device newBufferWithLength:sizeof(half) * M * K
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [device newBufferWithLength:sizeof(half) * K * N
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> c = [device newBufferWithLength:sizeof(half) * M * N
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> bias = [device newBufferWithLength:sizeof(half) * N
                                               options:MTLResourceStorageModeShared];
      if (a && b && c && bias) {
        fill_matrix((half *)[a contents], M, K, 0.01f);
        fill_matrix((half *)[b contents], K, N, 0.02f);
        memset([c contents], 0, sizeof(half) * M * N);
        memset([bias contents], 0, sizeof(half) * N);
        double t0 = now_sec();
        for (uint32_t i = 0; i < iters; ++i) {
          bwpp_metal_dispatch_matmul(device, queue, a, b, c, bias, params, mslSource);
        }
        double t1 = now_sec();
        double secs = t1 - t0;
        double flops = 2.0 * (double)M * (double)N * (double)K * (double)iters;
        double gflops = (flops / 1e9) / (secs > 0.0 ? secs : 1.0);
        printf("metal matmul: M=%u N=%u K=%u iters=%u time=%.6fs gflops=%.2f\n",
               M, N, K, iters, secs, gflops);
      }
    }

    if (!skip_softmax && has_kernel(mslC, "bwpp.meta: aux_kernel=softmax_f16")) {
      BwppSoftmaxParams params = { rows, cols, cols };
      id<MTLBuffer> x = [device newBufferWithLength:sizeof(half) * rows * cols
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> y = [device newBufferWithLength:sizeof(half) * rows * cols
                                            options:MTLResourceStorageModeShared];
      if (x && y) {
        fill_matrix((half *)[x contents], rows, cols, 0.01f);
        memset([y contents], 0, sizeof(half) * rows * cols);
        double t0 = now_sec();
        for (uint32_t i = 0; i < iters; ++i) {
          bwpp_metal_dispatch_softmax(device, queue, x, y, params, mslSource);
        }
        double t1 = now_sec();
        printf("metal softmax: rows=%u cols=%u iters=%u time=%.6fs\n",
               rows, cols, iters, t1 - t0);
      }
    }

    if (!skip_rmsnorm && has_kernel(mslC, "bwpp.meta: aux_kernel=rmsnorm_f16")) {
      BwppRmsnormParams params = { rows, cols, cols, 1e-5f };
      id<MTLBuffer> x = [device newBufferWithLength:sizeof(half) * rows * cols
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> y = [device newBufferWithLength:sizeof(half) * rows * cols
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> gamma = [device newBufferWithLength:sizeof(half) * cols
                                               options:MTLResourceStorageModeShared];
      if (x && y && gamma) {
        fill_matrix((half *)[x contents], rows, cols, 0.01f);
        memset([y contents], 0, sizeof(half) * rows * cols);
        memset([gamma contents], 0x3c, sizeof(half) * cols);
        double t0 = now_sec();
        for (uint32_t i = 0; i < iters; ++i) {
          bwpp_metal_dispatch_rmsnorm(device, queue, x, gamma, nil, y, params, mslSource);
        }
        double t1 = now_sec();
        printf("metal rmsnorm: rows=%u cols=%u iters=%u time=%.6fs\n",
               rows, cols, iters, t1 - t0);
      }
    }

    if (!skip_attention && has_kernel(mslC, "bwpp.meta: kernel=attention_f16")) {
      BwppAttentionParams params = { M, N, K, D, K, K, D, D };
      id<MTLBuffer> q = [device newBufferWithLength:sizeof(half) * M * K
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> k = [device newBufferWithLength:sizeof(half) * N * K
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> v = [device newBufferWithLength:sizeof(half) * N * D
                                            options:MTLResourceStorageModeShared];
      id<MTLBuffer> o = [device newBufferWithLength:sizeof(half) * M * D
                                            options:MTLResourceStorageModeShared];
      if (q && k && v && o) {
        fill_matrix((half *)[q contents], M, K, 0.03f);
        fill_matrix((half *)[k contents], N, K, 0.04f);
        fill_matrix((half *)[v contents], N, D, 0.02f);
        memset([o contents], 0, sizeof(half) * M * D);
        double t0 = now_sec();
        for (uint32_t i = 0; i < iters; ++i) {
          bwpp_metal_dispatch_attention(device, queue, q, k, v, o, params, mslSource);
        }
        double t1 = now_sec();
        double flops = 2.0 * (double)M * (double)N * (double)K + 2.0 * (double)M * (double)N * (double)D;
        flops *= (double)iters;
        double gflops = (flops / 1e9) / (t1 - t0 > 0.0 ? (t1 - t0) : 1.0);
        printf("metal attention: M=%u N=%u K=%u D=%u iters=%u time=%.6fs gflops=%.2f\n",
               M, N, K, D, iters, t1 - t0, gflops);
      }
    }
  }
  return 0;
}

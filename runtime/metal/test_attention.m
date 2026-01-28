#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>

#import "dispatch_stub.h"

typedef __fp16 half;

static void fill_matrix(half *dst, uint32_t rows, uint32_t cols, float scale) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      float v = (float)(i * cols + j + 1) * scale;
      dst[i * cols + j] = (half)v;
    }
  }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    NSString *mslPath = @"out_attention.metal";
    if (argc > 1) {
      mslPath = [NSString stringWithUTF8String:argv[1]];
    }
    NSError *err = nil;
    NSString *mslSource = [NSString stringWithContentsOfFile:mslPath
                                                   encoding:NSUTF8StringEncoding
                                                      error:&err];
    if (!mslSource) {
      fprintf(stderr, "failed to read MSL: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    const uint32_t M = 2;
    const uint32_t N = 3;
    const uint32_t K = 4;
    const uint32_t D = 5;

    BwppAttentionParams params = { M, N, K, D, K, K, D, D };

    id<MTLBuffer> q = [device newBufferWithLength:sizeof(half) * M * K
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> k = [device newBufferWithLength:sizeof(half) * N * K
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> v = [device newBufferWithLength:sizeof(half) * N * D
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> o = [device newBufferWithLength:sizeof(half) * M * D
                                          options:MTLResourceStorageModeShared];
    if (!q || !k || !v || !o) {
      fprintf(stderr, "failed to allocate buffers\n");
      return 1;
    }

    half *qPtr = (half *)[q contents];
    half *kPtr = (half *)[k contents];
    half *vPtr = (half *)[v contents];
    half *oPtr = (half *)[o contents];
    fill_matrix(qPtr, M, K, 0.03f);
    fill_matrix(kPtr, N, K, 0.04f);
    fill_matrix(vPtr, N, D, 0.02f);
    for (uint32_t i = 0; i < M * D; ++i) {
      oPtr[i] = (half)0.0f;
    }

    float ref[M * D];
    for (uint32_t i = 0; i < M * D; ++i) {
      ref[i] = 0.0f;
    }
    for (uint32_t m = 0; m < M; ++m) {
      float maxv = -INFINITY;
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (uint32_t kk = 0; kk < K; ++kk) {
          acc += (float)qPtr[m * K + kk] * (float)kPtr[n * K + kk];
        }
        if (acc > maxv) {
          maxv = acc;
        }
      }
      float sum = 0.0f;
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (uint32_t kk = 0; kk < K; ++kk) {
          acc += (float)qPtr[m * K + kk] * (float)kPtr[n * K + kk];
        }
        sum += expf(acc - maxv);
      }
      float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
      for (uint32_t d = 0; d < D; ++d) {
        float out = 0.0f;
        for (uint32_t n = 0; n < N; ++n) {
          float acc = 0.0f;
          for (uint32_t kk = 0; kk < K; ++kk) {
            acc += (float)qPtr[m * K + kk] * (float)kPtr[n * K + kk];
          }
          float w = expf(acc - maxv) * inv;
          out += w * (float)vPtr[n * D + d];
        }
        ref[m * D + d] = out;
      }
    }

    bwpp_metal_dispatch_attention(device, queue, q, k, v, o, params, mslSource);

    float maxErr = 0.0f;
    for (uint32_t i = 0; i < M * D; ++i) {
      float got = (float)oPtr[i];
      float diff = fabsf(got - ref[i]);
      if (diff > maxErr) {
        maxErr = diff;
      }
    }

    if (maxErr > 2e-2f) {
      fprintf(stderr, "FAIL attention max_err=%.6f\n", maxErr);
      return 1;
    }
    printf("METAL PASS attention max_err=%.6f\n", maxErr);
  }
  return 0;
}

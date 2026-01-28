#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

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

static float bwpp_silu(float x) {
  return x / (1.0f + expf(-x));
}

static void parse_epilogue(const char *src, int *ep_add, int *ep_silu) {
  *ep_add = 0;
  *ep_silu = 0;
  if (!src) {
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

int main(int argc, char **argv) {
  @autoreleasepool {
    NSString *mslPath = @"out.metal";
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
    const char *mslCStr = [mslSource UTF8String];
    int ep_add = 0;
    int ep_silu = 0;
    parse_epilogue(mslCStr, &ep_add, &ep_silu);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    const uint32_t M = 4;
    const uint32_t N = 4;
    const uint32_t K = 4;
    BwppMatmulParams params = { M, N, K, K, N, N };

    id<MTLBuffer> a = [device newBufferWithLength:sizeof(half) * M * K
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:sizeof(half) * K * N
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [device newBufferWithLength:sizeof(half) * M * N
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = [device newBufferWithLength:sizeof(half) * N
                                             options:MTLResourceStorageModeShared];
    if (!a || !b || !c || !bias) {
      fprintf(stderr, "failed to allocate buffers\n");
      return 1;
    }

    half *aPtr = (half *)[a contents];
    half *bPtr = (half *)[b contents];
    half *cPtr = (half *)[c contents];
    half *biasPtr = (half *)[bias contents];
    fill_matrix(aPtr, M, K, 0.1f);
    fill_matrix(bPtr, K, N, 0.05f);
    for (uint32_t i = 0; i < M * N; ++i) {
      cPtr[i] = (half)0.0f;
    }
    for (uint32_t i = 0; i < N; ++i) {
      float v = ep_add ? (0.01f * (float)(i + 1)) : 0.0f;
      biasPtr[i] = (half)v;
    }

    float ref[16];
    for (uint32_t i = 0; i < M * N; ++i) {
      ref[i] = 0.0f;
    }
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
          float av = (float)aPtr[row * K + k];
          float bv = (float)bPtr[k * N + col];
          acc += av * bv;
        }
        float out = acc;
        if (ep_add) {
          out += (float)biasPtr[col];
        }
        if (ep_silu) {
          out = bwpp_silu(out);
        }
        ref[row * N + col] = out;
      }
    }

    bwpp_metal_dispatch_matmul(device, queue, a, b, c, bias, params, mslSource);

    float maxErr = 0.0f;
    for (uint32_t i = 0; i < M * N; ++i) {
      float got = (float)cPtr[i];
      float diff = fabsf(got - ref[i]);
      if (diff > maxErr) {
        maxErr = diff;
      }
    }

    if (maxErr > 0.2f) {
      fprintf(stderr, "FAIL max_err=%.6f ep_add=%d ep_silu=%d\n", maxErr, ep_add, ep_silu);
      return 1;
    }
    printf("PASS max_err=%.6f ep_add=%d ep_silu=%d\n", maxErr, ep_add, ep_silu);
  }
  return 0;
}

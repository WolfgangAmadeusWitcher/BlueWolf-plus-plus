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
    NSString *mslPath = @"out_norm.metal";
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

    const uint32_t rows = 2;
    const uint32_t cols = 4;
    const uint32_t ld = cols;

    id<MTLBuffer> x = [device newBufferWithLength:sizeof(half) * rows * cols
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> y = [device newBufferWithLength:sizeof(half) * rows * cols
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> z = [device newBufferWithLength:sizeof(half) * rows * cols
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> gamma = [device newBufferWithLength:sizeof(half) * cols
                                              options:MTLResourceStorageModeShared];

    if (!x || !y || !z || !gamma) {
      fprintf(stderr, "failed to allocate buffers\n");
      return 1;
    }

    half *xPtr = (half *)[x contents];
    half *yPtr = (half *)[y contents];
    half *zPtr = (half *)[z contents];
    half *gPtr = (half *)[gamma contents];

    fill_matrix(xPtr, rows, cols, 0.1f);
    for (uint32_t i = 0; i < rows * cols; ++i) {
      yPtr[i] = (half)0.0f;
      zPtr[i] = (half)0.0f;
    }
    for (uint32_t i = 0; i < cols; ++i) {
      gPtr[i] = (half)1.0f;
    }

    BwppSoftmaxParams sparams = { rows, cols, ld };
    bwpp_metal_dispatch_softmax(device, queue, x, y, sparams, mslSource);

    for (uint32_t r = 0; r < rows; ++r) {
      float sum = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        sum += (float)yPtr[r * cols + c];
      }
      if (fabsf(sum - 1.0f) > 1e-2f) {
        fprintf(stderr, "softmax row sum mismatch: %.6f\n", sum);
        return 1;
      }
    }

    BwppRmsnormParams nparams = { rows, cols, ld, 1e-5f };
    bwpp_metal_dispatch_rmsnorm(device, queue, x, gamma, nil, z, nparams, mslSource);

    for (uint32_t r = 0; r < rows; ++r) {
      float sumsq = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float v = (float)zPtr[r * cols + c];
        sumsq += v * v;
      }
      float rms = sqrtf(sumsq / (float)cols);
      if (fabsf(rms - 1.0f) > 2e-2f) {
        fprintf(stderr, "rmsnorm rms mismatch: %.6f\n", rms);
        return 1;
      }
    }

    printf("METAL PASS softmax+rmsnorm\n");
  }
  return 0;
}

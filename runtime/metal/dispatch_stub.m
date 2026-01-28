#import "dispatch_stub.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static id<MTLLibrary> bwpp_build_library(id<MTLDevice> device, NSString *mslSource) {
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:mslSource options:nil error:&error];
  if (!library) {
    return nil;
  }
  return library;
}

static id<MTLComputePipelineState> bwpp_build_pipeline(id<MTLDevice> device,
                                                       id<MTLLibrary> library,
                                                       NSString *name) {
  NSError *error = nil;
  id<MTLFunction> func = [library newFunctionWithName:name];
  if (!func) {
    return nil;
  }
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
  return pso;
}

void bwpp_metal_dispatch_matmul(id<MTLDevice> device,
                                id<MTLCommandQueue> queue,
                                id<MTLBuffer> a,
                                id<MTLBuffer> b,
                                id<MTLBuffer> c,
                                id<MTLBuffer> bias,
                                BwppMatmulParams params,
                                NSString *mslSource) {
  if (!device || !queue || !a || !b || !c || !mslSource) {
    return;
  }

  const char *src = [mslSource UTF8String];
  if (!src) {
    return;
  }
  if (strstr(src, "bwpp.meta: kernel=none") != NULL) {
    return;
  }
  if (strstr(src, "bwpp.meta: kernel=matmul_f16") == NULL) {
    return;
  }

  uint32_t tileM = 0;
  uint32_t tileN = 0;
  uint32_t tileK = 0;
  const char *tilePtr = strstr(src, "bwpp.meta: tile=");
  if (tilePtr) {
    if (sscanf(tilePtr, "bwpp.meta: tile=%u,%u,%u", &tileM, &tileN, &tileK) != 3) {
      tileM = 0;
      tileN = 0;
      tileK = 0;
    }
  }

  id<MTLLibrary> library = bwpp_build_library(device, mslSource);
  if (!library) {
    return;
  }

  id<MTLComputePipelineState> pso = bwpp_build_pipeline(device, library, @"bwpp_matmul_f16");
  if (!pso) {
    return;
  }

  id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                length:sizeof(BwppMatmulParams)
                                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> biasBuf = bias;
  if (!biasBuf) {
    NSUInteger len = (NSUInteger)params.N * sizeof(uint16_t);
    biasBuf = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (biasBuf) {
      memset([biasBuf contents], 0, len);
    }
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:a offset:0 atIndex:0];
  [enc setBuffer:b offset:0 atIndex:1];
  [enc setBuffer:c offset:0 atIndex:2];
  [enc setBuffer:paramsBuf offset:0 atIndex:3];
  if (biasBuf) {
    [enc setBuffer:biasBuf offset:0 atIndex:4];
  }

  if (tileM == 0 || tileN == 0) {
    NSUInteger w = pso.threadExecutionWidth;
    MTLSize grid = MTLSizeMake(params.N, params.M, 1);
    MTLSize tg = MTLSizeMake(w, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  } else {
    NSUInteger total = pso.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)tileM * (NSUInteger)tileN > total) {
      return;
    }
    MTLSize tg = MTLSizeMake(tileN, tileM, 1);
    MTLSize grid = MTLSizeMake((params.N + tileN - 1) / tileN,
                               (params.M + tileM - 1) / tileM,
                               1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

void bwpp_metal_dispatch_softmax(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLBuffer> x,
                                 id<MTLBuffer> y,
                                 BwppSoftmaxParams params,
                                 NSString *mslSource) {
  if (!device || !queue || !x || !y || !mslSource) {
    return;
  }
  const char *src = [mslSource UTF8String];
  if (!src) {
    return;
  }
  if (strstr(src, "bwpp_softmax_f16") == NULL) {
    return;
  }

  id<MTLLibrary> library = bwpp_build_library(device, mslSource);
  if (!library) {
    return;
  }
  id<MTLComputePipelineState> pso = bwpp_build_pipeline(device, library, @"bwpp_softmax_f16");
  if (!pso) {
    return;
  }

  id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                length:sizeof(BwppSoftmaxParams)
                                               options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:x offset:0 atIndex:0];
  [enc setBuffer:y offset:0 atIndex:1];
  [enc setBuffer:paramsBuf offset:0 atIndex:2];

  MTLSize grid = MTLSizeMake(params.rows, 1, 1);
  NSUInteger w = pso.threadExecutionWidth;
  MTLSize tg = MTLSizeMake(w, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

void bwpp_metal_dispatch_rmsnorm(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLBuffer> x,
                                 id<MTLBuffer> gamma,
                                 id<MTLBuffer> beta,
                                 id<MTLBuffer> y,
                                 BwppRmsnormParams params,
                                 NSString *mslSource) {
  if (!device || !queue || !x || !y || !mslSource) {
    return;
  }
  const char *src = [mslSource UTF8String];
  if (!src) {
    return;
  }
  if (strstr(src, "bwpp_rmsnorm_f16") == NULL) {
    return;
  }

  id<MTLLibrary> library = bwpp_build_library(device, mslSource);
  if (!library) {
    return;
  }
  id<MTLComputePipelineState> pso = bwpp_build_pipeline(device, library, @"bwpp_rmsnorm_f16");
  if (!pso) {
    return;
  }

  id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                length:sizeof(BwppRmsnormParams)
                                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> gammaBuf = gamma;
  if (!gammaBuf) {
    NSUInteger len = (NSUInteger)params.cols * sizeof(uint16_t);
    gammaBuf = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (gammaBuf) {
      uint16_t *ptr = (uint16_t *)[gammaBuf contents];
      for (uint32_t i = 0; i < params.cols; ++i) {
        ptr[i] = 0x3c00; /* half(1.0) */
      }
    }
  }

  id<MTLBuffer> betaBuf = beta;
  if (!betaBuf) {
    NSUInteger len = (NSUInteger)params.cols * sizeof(uint16_t);
    betaBuf = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (betaBuf) {
      memset([betaBuf contents], 0, len);
    }
  }

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:x offset:0 atIndex:0];
  [enc setBuffer:gammaBuf offset:0 atIndex:1];
  [enc setBuffer:y offset:0 atIndex:2];
  [enc setBuffer:paramsBuf offset:0 atIndex:3];
  if (betaBuf) {
    [enc setBuffer:betaBuf offset:0 atIndex:4];
  }

  MTLSize grid = MTLSizeMake(params.rows, 1, 1);
  NSUInteger w = pso.threadExecutionWidth;
  MTLSize tg = MTLSizeMake(w, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

void bwpp_metal_dispatch_attention(id<MTLDevice> device,
                                   id<MTLCommandQueue> queue,
                                   id<MTLBuffer> q,
                                   id<MTLBuffer> k,
                                   id<MTLBuffer> v,
                                   id<MTLBuffer> o,
                                   BwppAttentionParams params,
                                   NSString *mslSource) {
  if (!device || !queue || !q || !k || !v || !o || !mslSource) {
    return;
  }
  const char *src = [mslSource UTF8String];
  if (!src) {
    return;
  }
  if (strstr(src, "bwpp.meta: kernel=attention_f16") == NULL) {
    return;
  }

  uint32_t tileM = 0;
  uint32_t tileN = 0;
  uint32_t tileK = 0;
  const char *tilePtr = strstr(src, "bwpp.meta: tile=");
  if (tilePtr) {
    if (sscanf(tilePtr, "bwpp.meta: tile=%u,%u,%u", &tileM, &tileN, &tileK) != 3) {
      tileM = 0;
      tileN = 0;
      tileK = 0;
    }
  }

  id<MTLLibrary> library = bwpp_build_library(device, mslSource);
  if (!library) {
    return;
  }

  id<MTLComputePipelineState> pso = bwpp_build_pipeline(device, library, @"bwpp_attention_f16");
  if (!pso) {
    return;
  }

  id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                length:sizeof(BwppAttentionParams)
                                               options:MTLResourceStorageModeShared];

  uint32_t tile = tileM;
  if (tile == 0) {
    NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger side = (NSUInteger)sqrt((double)maxThreads);
    if (side == 0) {
      side = 1;
    }
    tile = (uint32_t)side;
  }
  NSUInteger total = pso.maxTotalThreadsPerThreadgroup;
  if ((NSUInteger)tile * (NSUInteger)tile > total) {
    return;
  }

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:q offset:0 atIndex:0];
  [enc setBuffer:k offset:0 atIndex:1];
  [enc setBuffer:v offset:0 atIndex:2];
  [enc setBuffer:o offset:0 atIndex:3];
  [enc setBuffer:paramsBuf offset:0 atIndex:4];

  MTLSize tg = MTLSizeMake(tile, tile, 1);
  MTLSize grid = MTLSizeMake((params.D + tile - 1) / tile,
                             (params.M + tile - 1) / tile,
                             1);
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

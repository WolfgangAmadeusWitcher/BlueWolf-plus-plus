#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
} BwppMatmulParams;

typedef struct {
  uint32_t rows;
  uint32_t cols;
  uint32_t ld;
} BwppSoftmaxParams;

typedef struct {
  uint32_t rows;
  uint32_t cols;
  uint32_t ld;
  float eps;
} BwppRmsnormParams;

typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t D;
  uint32_t ldq;
  uint32_t ldk;
  uint32_t ldv;
  uint32_t ldo;
} BwppAttentionParams;

void bwpp_metal_dispatch_matmul(id<MTLDevice> device,
                                id<MTLCommandQueue> queue,
                                id<MTLBuffer> a,
                                id<MTLBuffer> b,
                                id<MTLBuffer> c,
                                id<MTLBuffer> bias,
                                BwppMatmulParams params,
                                NSString *mslSource);

void bwpp_metal_dispatch_softmax(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLBuffer> x,
                                 id<MTLBuffer> y,
                                 BwppSoftmaxParams params,
                                 NSString *mslSource);

void bwpp_metal_dispatch_rmsnorm(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLBuffer> x,
                                 id<MTLBuffer> gamma,
                                 id<MTLBuffer> beta,
                                 id<MTLBuffer> y,
                                 BwppRmsnormParams params,
                                 NSString *mslSource);

void bwpp_metal_dispatch_attention(id<MTLDevice> device,
                                   id<MTLCommandQueue> queue,
                                   id<MTLBuffer> q,
                                   id<MTLBuffer> k,
                                   id<MTLBuffer> v,
                                   id<MTLBuffer> o,
                                   BwppAttentionParams params,
                                   NSString *mslSource);

#ifdef __cplusplus
}
#endif

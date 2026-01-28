#include "tile_ir.h"
#include <stdlib.h>

BwppTileKernel *bwpp_tile_kernel_create(void) {
  BwppTileKernel *kernel = (BwppTileKernel *)calloc(1, sizeof(BwppTileKernel));
  return kernel;
}

static int bwpp_tile_kernel_grow(BwppTileKernel *kernel) {
  uint32_t next = kernel->op_capacity ? kernel->op_capacity * 2u : 4u;
  BwppTileOp *ops = (BwppTileOp *)realloc(kernel->ops, next * sizeof(BwppTileOp));
  if (!ops) {
    return 0;
  }
  kernel->ops = ops;
  kernel->op_capacity = next;
  return 1;
}

BwppStatus bwpp_tile_kernel_add_op(BwppTileKernel *kernel, const BwppTileOp *op) {
  if (!kernel || !op) {
    return BWPP_ERR;
  }
  if (kernel->op_count >= kernel->op_capacity && !bwpp_tile_kernel_grow(kernel)) {
    return BWPP_ERR;
  }
  kernel->ops[kernel->op_count++] = *op;
  return BWPP_OK;
}

void bwpp_tile_kernel_destroy(BwppTileKernel *kernel) {
  if (!kernel) {
    return;
  }
  free(kernel->ops);
  free(kernel);
}

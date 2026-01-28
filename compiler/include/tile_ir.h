#ifndef BWPP_TILE_IR_H
#define BWPP_TILE_IR_H

#include "bwpp.h"
#include <stdint.h>

typedef enum {
  BWPP_TILE_AXIS_M = 0,
  BWPP_TILE_AXIS_N,
  BWPP_TILE_AXIS_K
} BwppTileAxis;

typedef enum {
  BWPP_TILE_MEM_GLOBAL = 0,
  BWPP_TILE_MEM_THREADGROUP,
  BWPP_TILE_MEM_REGISTER
} BwppTileMemory;

typedef enum {
  BWPP_TILE_OP_MATMUL = 0,
  BWPP_TILE_OP_LOAD,
  BWPP_TILE_OP_STORE,
  BWPP_TILE_OP_ELEMENTWISE,
  BWPP_TILE_OP_ATTENTION
} BwppTileOpKind;

typedef enum {
  BWPP_TILE_ROLE_NONE = 0,
  BWPP_TILE_ROLE_A,
  BWPP_TILE_ROLE_B,
  BWPP_TILE_ROLE_C
} BwppTileRole;

typedef enum {
  BWPP_TILE_EPILOGUE_NONE = 0,
  BWPP_TILE_EPILOGUE_ADD,
  BWPP_TILE_EPILOGUE_SILU,
  BWPP_TILE_EPILOGUE_ADD_SILU
} BwppTileEpilogue;

typedef struct {
  uint32_t m;
  uint32_t n;
  uint32_t k;
} BwppTileShape;

typedef struct {
  BwppTileOpKind kind;
  BwppTileShape tile;
  BwppTileMemory a_mem;
  BwppTileMemory b_mem;
  BwppTileMemory c_mem;
  BwppTileEpilogue epilogue;
  BwppTileMemory src_mem;
  BwppTileMemory dst_mem;
  BwppTileRole role;
} BwppTileOp;

typedef struct {
  BwppTileOp *ops;
  uint32_t op_count;
  uint32_t op_capacity;
  BwppTileShape block;
} BwppTileKernel;

BwppTileKernel *bwpp_tile_kernel_create(void);
void bwpp_tile_kernel_destroy(BwppTileKernel *kernel);
BwppStatus bwpp_tile_kernel_add_op(BwppTileKernel *kernel, const BwppTileOp *op);

#endif

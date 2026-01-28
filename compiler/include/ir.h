#ifndef BWPP_IR_H
#define BWPP_IR_H

#include "bwpp.h"
#include "ast.h"
#include <stdint.h>
#include <stdio.h>

typedef enum {
  BWPP_OP_MATMUL = 0,
  BWPP_OP_BATCH_MATMUL,
  BWPP_OP_TRANSPOSE,
  BWPP_OP_PERMUTE,
  BWPP_OP_RESHAPE,
  BWPP_OP_ADD,
  BWPP_OP_SUB,
  BWPP_OP_MUL,
  BWPP_OP_DIV,
  BWPP_OP_REDUCE_SUM,
  BWPP_OP_REDUCE_MAX,
  BWPP_OP_SOFTMAX,
  BWPP_OP_RMSNORM,
  BWPP_OP_SILU
} BwppOpKind;

typedef enum {
  BWPP_REGION_NORMAL = 0,
  BWPP_REGION_REVERSIBLE
} BwppRegionKind;

typedef enum {
  BWPP_POLICY_STORE = 0,
  BWPP_POLICY_RECOMPUTE,
  BWPP_POLICY_AUTO
} BwppRegionPolicy;

typedef struct {
  uint32_t id;
  BwppRegionKind kind;
  BwppRegionPolicy policy;
} BwppIrRegion;

typedef struct {
  BwppOpKind op;
  uint32_t region_id;
  uint32_t flags;
} BwppIrNode;

typedef struct {
  BwppIrNode *nodes;
  uint32_t node_count;
  uint32_t node_capacity;
  BwppIrRegion *regions;
  uint32_t region_count;
  uint32_t region_capacity;
  uint32_t flags;
} BwppIrModule;

enum { BWPP_IR_NO_REGION = 0xffffffffu };
enum { BWPP_IR_OPF_HAS_BIAS = 1u << 0 };
enum { BWPP_IRF_HAS_ATTENTION = 1u << 0 };

BwppIrModule *bwpp_ir_create(void);
BwppIrModule *bwpp_ir_from_ast(const BwppAstModule *module);
void bwpp_ir_destroy(BwppIrModule *ir);
uint32_t bwpp_ir_add_region(BwppIrModule *ir, BwppRegionKind kind, BwppRegionPolicy policy);
BwppStatus bwpp_ir_add_node(BwppIrModule *ir, BwppOpKind op, uint32_t region_id, uint32_t flags);
void bwpp_ir_dump(const BwppIrModule *ir, FILE *out);

#endif

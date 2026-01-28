#ifndef BWPP_AST_H
#define BWPP_AST_H

#include "bwpp.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
  BWPP_AST_MODULE = 0,
  BWPP_AST_FN,
  BWPP_AST_CALL
} BwppAstKind;

typedef struct BwppAstNode {
  BwppAstKind kind;
  const char *name;
} BwppAstNode;

typedef enum {
  BWPP_AST_OP_MATMUL = 0,
  BWPP_AST_OP_BATCH_MATMUL,
  BWPP_AST_OP_TRANSPOSE,
  BWPP_AST_OP_PERMUTE,
  BWPP_AST_OP_RESHAPE,
  BWPP_AST_OP_ADD,
  BWPP_AST_OP_SUB,
  BWPP_AST_OP_MUL,
  BWPP_AST_OP_DIV,
  BWPP_AST_OP_REDUCE_SUM,
  BWPP_AST_OP_REDUCE_MAX,
  BWPP_AST_OP_SOFTMAX,
  BWPP_AST_OP_RMSNORM,
  BWPP_AST_OP_SILU
} BwppAstOpKind;

typedef enum {
  BWPP_AST_REGION_NORMAL = 0,
  BWPP_AST_REGION_REVERSIBLE
} BwppAstRegionKind;

typedef enum {
  BWPP_AST_POLICY_STORE = 0,
  BWPP_AST_POLICY_RECOMPUTE,
  BWPP_AST_POLICY_AUTO
} BwppAstRegionPolicy;

typedef struct {
  BwppAstOpKind op;
  uint32_t region_id;
  uint32_t flags;
} BwppAstOp;

typedef struct {
  uint32_t id;
  BwppAstRegionKind kind;
  BwppAstRegionPolicy policy;
} BwppAstRegion;

typedef struct {
  BwppAstNode *root;
  BwppAstOp *ops;
  uint32_t op_count;
  uint32_t op_capacity;
  BwppAstRegion *regions;
  uint32_t region_count;
  uint32_t region_capacity;
  const char *source;
  size_t length;
} BwppAstModule;

BwppAstModule *bwpp_ast_module_create(const char *source, size_t length);
void bwpp_ast_module_destroy(BwppAstModule *module);
BwppStatus bwpp_ast_add_op(BwppAstModule *module, BwppAstOpKind op, uint32_t region_id, uint32_t flags);
uint32_t bwpp_ast_add_region(BwppAstModule *module, BwppAstRegionKind kind, BwppAstRegionPolicy policy);

enum { BWPP_AST_NO_REGION = 0xffffffffu };
enum { BWPP_AST_OPF_HAS_BIAS = 1u << 0 };

#endif

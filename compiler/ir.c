#include "ir.h"
#include "graph_ir.h"
#include <stdlib.h>
#include <string.h>

BwppIrModule *bwpp_ir_create(void) {
  BwppIrModule *ir = (BwppIrModule *)calloc(1, sizeof(BwppIrModule));
  return ir;
}

static int bwpp_ir_grow_nodes(BwppIrModule *ir) {
  uint32_t next = ir->node_capacity ? ir->node_capacity * 2u : 8u;
  BwppIrNode *nodes = (BwppIrNode *)realloc(ir->nodes, next * sizeof(BwppIrNode));
  if (!nodes) {
    return 0;
  }
  ir->nodes = nodes;
  ir->node_capacity = next;
  return 1;
}

static int bwpp_ir_grow_regions(BwppIrModule *ir) {
  uint32_t next = ir->region_capacity ? ir->region_capacity * 2u : 4u;
  BwppIrRegion *regions = (BwppIrRegion *)realloc(ir->regions, next * sizeof(BwppIrRegion));
  if (!regions) {
    return 0;
  }
  ir->regions = regions;
  ir->region_capacity = next;
  return 1;
}

uint32_t bwpp_ir_add_region(BwppIrModule *ir, BwppRegionKind kind, BwppRegionPolicy policy) {
  if (!ir) {
    return BWPP_IR_NO_REGION;
  }
  if (ir->region_count >= ir->region_capacity && !bwpp_ir_grow_regions(ir)) {
    return BWPP_IR_NO_REGION;
  }
  uint32_t id = ir->region_count;
  BwppIrRegion region;
  region.id = id;
  region.kind = kind;
  region.policy = policy;
  ir->regions[ir->region_count++] = region;
  return id;
}

BwppStatus bwpp_ir_add_node(BwppIrModule *ir, BwppOpKind op, uint32_t region_id, uint32_t flags) {
  if (!ir) {
    return BWPP_ERR;
  }
  if (ir->node_count >= ir->node_capacity && !bwpp_ir_grow_nodes(ir)) {
    return BWPP_ERR;
  }
  BwppIrNode node;
  node.op = op;
  node.region_id = region_id;
  node.flags = flags;
  ir->nodes[ir->node_count++] = node;
  return BWPP_OK;
}

static BwppOpKind bwpp_map_op(BwppAstOpKind op) {
  switch (op) {
    case BWPP_AST_OP_MATMUL:
      return BWPP_OP_MATMUL;
    case BWPP_AST_OP_BATCH_MATMUL:
      return BWPP_OP_BATCH_MATMUL;
    case BWPP_AST_OP_TRANSPOSE:
      return BWPP_OP_TRANSPOSE;
    case BWPP_AST_OP_PERMUTE:
      return BWPP_OP_PERMUTE;
    case BWPP_AST_OP_RESHAPE:
      return BWPP_OP_RESHAPE;
    case BWPP_AST_OP_ADD:
      return BWPP_OP_ADD;
    case BWPP_AST_OP_SUB:
      return BWPP_OP_SUB;
    case BWPP_AST_OP_MUL:
      return BWPP_OP_MUL;
    case BWPP_AST_OP_DIV:
      return BWPP_OP_DIV;
    case BWPP_AST_OP_REDUCE_SUM:
      return BWPP_OP_REDUCE_SUM;
    case BWPP_AST_OP_REDUCE_MAX:
      return BWPP_OP_REDUCE_MAX;
    case BWPP_AST_OP_SOFTMAX:
      return BWPP_OP_SOFTMAX;
    case BWPP_AST_OP_RMSNORM:
      return BWPP_OP_RMSNORM;
    case BWPP_AST_OP_SILU:
      return BWPP_OP_SILU;
  }
  return BWPP_OP_MATMUL;
}

static BwppRegionKind bwpp_map_region_kind(BwppAstRegionKind kind) {
  return kind == BWPP_AST_REGION_REVERSIBLE ? BWPP_REGION_REVERSIBLE : BWPP_REGION_NORMAL;
}

static BwppRegionPolicy bwpp_map_region_policy(BwppAstRegionPolicy policy) {
  switch (policy) {
    case BWPP_AST_POLICY_STORE:
      return BWPP_POLICY_STORE;
    case BWPP_AST_POLICY_RECOMPUTE:
      return BWPP_POLICY_RECOMPUTE;
    case BWPP_AST_POLICY_AUTO:
      return BWPP_POLICY_AUTO;
  }
  return BWPP_POLICY_AUTO;
}

static uint32_t bwpp_map_flags(uint32_t flags) {
  uint32_t out = 0;
  if (flags & BWPP_AST_OPF_HAS_BIAS) {
    out |= BWPP_IR_OPF_HAS_BIAS;
  }
  return out;
}

static int bwpp_map_graph_op(BwppGraphOpKind op, BwppOpKind *out) {
  switch (op) {
    case BWPP_GOP_MATMUL:
      *out = BWPP_OP_MATMUL;
      return 1;
    case BWPP_GOP_BATCH_MATMUL:
      *out = BWPP_OP_BATCH_MATMUL;
      return 1;
    case BWPP_GOP_TRANSPOSE:
      *out = BWPP_OP_TRANSPOSE;
      return 1;
    case BWPP_GOP_PERMUTE:
      *out = BWPP_OP_PERMUTE;
      return 1;
    case BWPP_GOP_RESHAPE:
      *out = BWPP_OP_RESHAPE;
      return 1;
    case BWPP_GOP_ADD:
      *out = BWPP_OP_ADD;
      return 1;
    case BWPP_GOP_SUB:
      *out = BWPP_OP_SUB;
      return 1;
    case BWPP_GOP_MUL:
      *out = BWPP_OP_MUL;
      return 1;
    case BWPP_GOP_DIV:
      *out = BWPP_OP_DIV;
      return 1;
    case BWPP_GOP_REDUCE_SUM:
      *out = BWPP_OP_REDUCE_SUM;
      return 1;
    case BWPP_GOP_REDUCE_MAX:
      *out = BWPP_OP_REDUCE_MAX;
      return 1;
    case BWPP_GOP_SOFTMAX:
      *out = BWPP_OP_SOFTMAX;
      return 1;
    case BWPP_GOP_RMSNORM:
      *out = BWPP_OP_RMSNORM;
      return 1;
    case BWPP_GOP_SILU:
      *out = BWPP_OP_SILU;
      return 1;
    default:
      return 0;
  }
}

static uint32_t bwpp_map_graph_flags(uint32_t flags) {
  uint32_t out = 0;
  if (flags & BWPP_GRAPH_OPF_HAS_BIAS) {
    out |= BWPP_IR_OPF_HAS_BIAS;
  }
  return out;
}

BwppIrModule *bwpp_ir_from_ast(const BwppAstModule *module) {
  BwppIrModule *ir = bwpp_ir_create();
  if (!ir || !module) {
    return ir;
  }

  uint32_t *region_map = NULL;
  if (module->region_count > 0) {
    region_map = (uint32_t *)calloc(module->region_count, sizeof(uint32_t));
  }

  for (uint32_t i = 0; i < module->region_count; ++i) {
    const BwppAstRegion *r = &module->regions[i];
    uint32_t id = bwpp_ir_add_region(ir, bwpp_map_region_kind(r->kind), bwpp_map_region_policy(r->policy));
    if (region_map) {
      region_map[i] = id;
    }
  }

  for (uint32_t i = 0; i < module->op_count; ++i) {
    const BwppAstOp *op = &module->ops[i];
    uint32_t region_id = BWPP_IR_NO_REGION;
    if (op->region_id != BWPP_AST_NO_REGION && region_map && op->region_id < module->region_count) {
      region_id = region_map[op->region_id];
    }
    bwpp_ir_add_node(ir, bwpp_map_op(op->op), region_id, bwpp_map_flags(op->flags));
  }

  free(region_map);
  return ir;
}

BwppIrModule *bwpp_ir_from_graph(const BwppGraph *graph) {
  BwppIrModule *ir = bwpp_ir_create();
  if (!ir || !graph) {
    return ir;
  }

  uint32_t *region_map = NULL;
  if (graph->region_count > 0) {
    region_map = (uint32_t *)calloc(graph->region_count, sizeof(uint32_t));
  }

  for (uint32_t i = 0; i < graph->region_count; ++i) {
    const BwppGraphRegion *r = &graph->regions[i];
    uint32_t id = bwpp_ir_add_region(ir, r->kind, r->policy);
    if (region_map) {
      region_map[i] = id;
    }
  }

  for (uint32_t i = 0; i < graph->node_count; ++i) {
    const BwppGraphNode *node = &graph->nodes[i];
    BwppOpKind op;
    if (!bwpp_map_graph_op(node->op, &op)) {
      continue;
    }
    uint32_t region_id = BWPP_IR_NO_REGION;
    if (node->region_id != BWPP_GRAPH_NO_REGION && region_map &&
        node->region_id < graph->region_count) {
      region_id = region_map[node->region_id];
    }
    bwpp_ir_add_node(ir, op, region_id, bwpp_map_graph_flags(node->flags));
  }

  free(region_map);
  return ir;
}

void bwpp_ir_dump(const BwppIrModule *ir, FILE *out) {
  if (!ir || !out) {
    return;
  }
  fprintf(out, "ir.nodes=%u\n", ir->node_count);
  fprintf(out, "ir.regions=%u\n", ir->region_count);
  for (uint32_t i = 0; i < ir->region_count; ++i) {
    const BwppIrRegion *r = &ir->regions[i];
    fprintf(out, "region[%u]=kind:%u policy:%u\n", r->id, (unsigned)r->kind, (unsigned)r->policy);
  }
}

void bwpp_ir_destroy(BwppIrModule *ir) {
  if (!ir) {
    return;
  }
  free(ir->nodes);
  free(ir->regions);
  memset(ir, 0, sizeof(*ir));
  free(ir);
}

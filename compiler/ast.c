#include "ast.h"
#include <stdlib.h>

BwppAstModule *bwpp_ast_module_create(const char *source, size_t length) {
  BwppAstModule *module = (BwppAstModule *)calloc(1, sizeof(BwppAstModule));
  if (!module) {
    return NULL;
  }
  module->source = source;
  module->length = length;
  return module;
}

void bwpp_ast_module_destroy(BwppAstModule *module) {
  if (!module) {
    return;
  }
  free(module->ops);
  free(module->regions);
  free(module);
}

static int bwpp_ast_grow_ops(BwppAstModule *module) {
  uint32_t next = module->op_capacity ? module->op_capacity * 2u : 8u;
  BwppAstOp *ops = (BwppAstOp *)realloc(module->ops, next * sizeof(BwppAstOp));
  if (!ops) {
    return 0;
  }
  module->ops = ops;
  module->op_capacity = next;
  return 1;
}

static int bwpp_ast_grow_regions(BwppAstModule *module) {
  uint32_t next = module->region_capacity ? module->region_capacity * 2u : 4u;
  BwppAstRegion *regions = (BwppAstRegion *)realloc(module->regions, next * sizeof(BwppAstRegion));
  if (!regions) {
    return 0;
  }
  module->regions = regions;
  module->region_capacity = next;
  return 1;
}

uint32_t bwpp_ast_add_region(BwppAstModule *module, BwppAstRegionKind kind, BwppAstRegionPolicy policy) {
  if (!module) {
    return BWPP_AST_NO_REGION;
  }
  if (module->region_count >= module->region_capacity && !bwpp_ast_grow_regions(module)) {
    return BWPP_AST_NO_REGION;
  }
  uint32_t id = module->region_count;
  BwppAstRegion region;
  region.id = id;
  region.kind = kind;
  region.policy = policy;
  module->regions[module->region_count++] = region;
  return id;
}

BwppStatus bwpp_ast_add_op(BwppAstModule *module, BwppAstOpKind op, uint32_t region_id, uint32_t flags) {
  if (!module) {
    return BWPP_ERR;
  }
  if (module->op_count >= module->op_capacity && !bwpp_ast_grow_ops(module)) {
    return BWPP_ERR;
  }
  BwppAstOp entry;
  entry.op = op;
  entry.region_id = region_id;
  entry.flags = flags;
  module->ops[module->op_count++] = entry;
  return BWPP_OK;
}

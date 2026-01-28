#ifndef BWPP_GRAPH_IR_H
#define BWPP_GRAPH_IR_H

#include "bwpp.h"
#include "ir.h"
#include "ast.h"
#include <stdint.h>
#include <stdio.h>

#define BWPP_GRAPH_MAX_DIMS 4
#define BWPP_GRAPH_MAX_INPUTS 4

typedef struct {
  const char *ptr;
  size_t len;
} BwppStr;

typedef enum {
  BWPP_DTYPE_UNKNOWN = 0,
  BWPP_DTYPE_F16,
  BWPP_DTYPE_BF16,
  BWPP_DTYPE_F32
} BwppDType;

typedef enum {
  BWPP_LAYOUT_UNKNOWN = 0,
  BWPP_LAYOUT_ROW_MAJOR,
  BWPP_LAYOUT_COL_MAJOR
} BwppLayout;

typedef struct {
  uint32_t rank;
  BwppStr dims[BWPP_GRAPH_MAX_DIMS];
} BwppShape;

typedef enum {
  BWPP_GOP_MATMUL = 0,
  BWPP_GOP_BATCH_MATMUL,
  BWPP_GOP_TRANSPOSE,
  BWPP_GOP_PERMUTE,
  BWPP_GOP_RESHAPE,
  BWPP_GOP_ADD,
  BWPP_GOP_SUB,
  BWPP_GOP_MUL,
  BWPP_GOP_DIV,
  BWPP_GOP_REDUCE_SUM,
  BWPP_GOP_REDUCE_MAX,
  BWPP_GOP_SOFTMAX,
  BWPP_GOP_RMSNORM,
  BWPP_GOP_SILU,
  BWPP_GOP_SILU_GRAD,
  BWPP_GOP_SOFTMAX_GRAD,
  BWPP_GOP_RMSNORM_GRAD
} BwppGraphOpKind;

typedef struct {
  int has_axis;
  int axis;
  int has_epsilon;
  float epsilon;
  BwppShape shape;
  uint32_t perm[BWPP_GRAPH_MAX_DIMS];
  uint32_t perm_rank;
} BwppGraphAttr;

typedef struct {
  uint32_t id;
  BwppGraphOpKind op;
  uint32_t inputs[BWPP_GRAPH_MAX_INPUTS];
  uint32_t input_count;
  uint32_t output;
  BwppGraphAttr attr;
  uint32_t region_id;
  uint32_t flags;
} BwppGraphNode;

typedef struct {
  uint32_t id;
  BwppStr name;
  BwppDType dtype;
  BwppShape shape;
  BwppLayout layout;
  uint32_t producer;
  uint32_t flags;
} BwppGraphValue;

typedef struct {
  uint32_t id;
  BwppRegionKind kind;
  BwppRegionPolicy policy;
} BwppGraphRegion;

typedef struct {
  BwppGraphNode *nodes;
  uint32_t node_count;
  uint32_t node_capacity;
  BwppGraphValue *values;
  uint32_t value_count;
  uint32_t value_capacity;
  BwppGraphRegion *regions;
  uint32_t region_count;
  uint32_t region_capacity;
  uint32_t *outputs;
  uint32_t output_count;
  uint32_t output_capacity;
} BwppGraph;

enum { BWPP_GRAPH_NO_NODE = 0xffffffffu };
enum { BWPP_GRAPH_NO_VALUE = 0xffffffffu };
enum { BWPP_GRAPH_NO_REGION = 0xffffffffu };

enum {
  BWPP_GRAPH_VALUE_INPUT = 1u << 0,
  BWPP_GRAPH_VALUE_OUTPUT = 1u << 1,
  BWPP_GRAPH_VALUE_CONST = 1u << 2
};

enum { BWPP_GRAPH_OPF_HAS_BIAS = 1u << 0 };

BwppGraph *bwpp_graph_build(const BwppAstModule *module);
BwppGraph *bwpp_graph_autodiff(const BwppGraph *graph);
void bwpp_graph_destroy(BwppGraph *graph);
void bwpp_graph_dump(const BwppGraph *graph, FILE *out);
void bwpp_graph_dump_dot(const BwppGraph *graph, FILE *out);

#endif

#include "graph_ir.h"
#include "lexer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  BwppLexer lx;
  BwppToken lookahead;
  int has_lookahead;
} BwppGraphParser;

typedef struct {
  BwppStr name;
  uint32_t value_id;
} BwppBinding;

typedef struct {
  BwppBinding *bindings;
  uint32_t binding_count;
  uint32_t binding_capacity;
  BwppGraph *graph;
} BwppGraphBuilder;

static BwppToken bwpp_graph_next(BwppGraphParser *p) {
  if (p->has_lookahead) {
    p->has_lookahead = 0;
    return p->lookahead;
  }
  return bwpp_lexer_next(&p->lx);
}

static void bwpp_graph_unread(BwppGraphParser *p, BwppToken tok) {
  p->lookahead = tok;
  p->has_lookahead = 1;
}

static BwppStr bwpp_tok_str(const BwppToken *tok) {
  BwppStr out = { tok->lexeme, tok->length };
  return out;
}

static int bwpp_str_eq(BwppStr s, const char *lit) {
  size_t len = strlen(lit);
  return s.len == len && strncmp(s.ptr, lit, len) == 0;
}

static int bwpp_str_eq_str(BwppStr a, BwppStr b) {
  return a.len == b.len && strncmp(a.ptr, b.ptr, a.len) == 0;
}

static int bwpp_str_is_number(BwppStr s) {
  if (s.len == 0) {
    return 0;
  }
  for (size_t i = 0; i < s.len; ++i) {
    if (s.ptr[i] < '0' || s.ptr[i] > '9') {
      return 0;
    }
  }
  return 1;
}

static int bwpp_str_to_u32(BwppStr s, uint32_t *out) {
  if (!bwpp_str_is_number(s)) {
    return 0;
  }
  uint32_t value = 0;
  for (size_t i = 0; i < s.len; ++i) {
    value = value * 10u + (uint32_t)(s.ptr[i] - '0');
  }
  *out = value;
  return 1;
}

static int bwpp_str_to_f32(BwppStr s, float *out) {
  if (s.len == 0) {
    return 0;
  }
  char *tmp = (char *)malloc(s.len + 1);
  if (!tmp) {
    return 0;
  }
  memcpy(tmp, s.ptr, s.len);
  tmp[s.len] = '\0';
  char *end = NULL;
  double val = strtod(tmp, &end);
  int ok = (end && end != tmp);
  if (ok) {
    *out = (float)val;
  }
  free(tmp);
  return ok;
}

static int bwpp_tok_is(const BwppToken *tok, const char *lit) {
  size_t len = strlen(lit);
  return tok->kind == BWPP_TOK_IDENT && tok->length == len && strncmp(tok->lexeme, lit, len) == 0;
}

static BwppDType bwpp_parse_dtype(BwppStr s) {
  if (bwpp_str_eq(s, "f16")) {
    return BWPP_DTYPE_F16;
  }
  if (bwpp_str_eq(s, "bf16")) {
    return BWPP_DTYPE_BF16;
  }
  if (bwpp_str_eq(s, "f32")) {
    return BWPP_DTYPE_F32;
  }
  return BWPP_DTYPE_UNKNOWN;
}

static BwppLayout bwpp_parse_layout(BwppStr s) {
  if (bwpp_str_eq(s, "row_major")) {
    return BWPP_LAYOUT_ROW_MAJOR;
  }
  if (bwpp_str_eq(s, "col_major")) {
    return BWPP_LAYOUT_COL_MAJOR;
  }
  return BWPP_LAYOUT_UNKNOWN;
}

static int bwpp_parse_shape_list(BwppGraphParser *p, BwppShape *shape) {
  BwppToken tok = bwpp_graph_next(p);
  if (!(tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '[')) {
    return 0;
  }
  shape->rank = 0;
  for (;;) {
    BwppToken t = bwpp_graph_next(p);
    if (t.kind == BWPP_TOK_EOF) {
      return 0;
    }
    if (t.kind == BWPP_TOK_SYMBOL && t.length == 1) {
      if (t.lexeme[0] == ']') {
        break;
      }
      if (t.lexeme[0] == ',') {
        continue;
      }
    }
    if (shape->rank < BWPP_GRAPH_MAX_DIMS &&
        (t.kind == BWPP_TOK_IDENT || t.kind == BWPP_TOK_NUMBER)) {
      shape->dims[shape->rank++] = bwpp_tok_str(&t);
    }
  }
  return 1;
}

static void bwpp_shape_copy(BwppShape *dst, const BwppShape *src) {
  dst->rank = src->rank;
  for (uint32_t i = 0; i < src->rank && i < BWPP_GRAPH_MAX_DIMS; ++i) {
    dst->dims[i] = src->dims[i];
  }
}

static BwppShape bwpp_shape_broadcast(const BwppShape *a, const BwppShape *b) {
  BwppShape out = {0};
  uint32_t rank = a->rank > b->rank ? a->rank : b->rank;
  out.rank = rank;
  for (uint32_t i = 0; i < rank; ++i) {
    uint32_t ai = (a->rank > i) ? (a->rank - 1 - i) : UINT32_MAX;
    uint32_t bi = (b->rank > i) ? (b->rank - 1 - i) : UINT32_MAX;
    BwppStr ad = (ai != UINT32_MAX) ? a->dims[ai] : (BwppStr){ "1", 1 };
    BwppStr bd = (bi != UINT32_MAX) ? b->dims[bi] : (BwppStr){ "1", 1 };
    BwppStr chosen = ad;
    if (bwpp_str_eq(ad, "1")) {
      chosen = bd;
    } else if (bwpp_str_eq(bd, "1")) {
      chosen = ad;
    } else if (bwpp_str_eq_str(ad, bd)) {
      chosen = ad;
    }
    out.dims[rank - 1 - i] = chosen;
  }
  return out;
}

static int bwpp_shape_equal(const BwppShape *a, const BwppShape *b) {
  if (a->rank != b->rank) {
    return 0;
  }
  for (uint32_t i = 0; i < a->rank; ++i) {
    if (!bwpp_str_eq_str(a->dims[i], b->dims[i])) {
      return 0;
    }
  }
  return 1;
}

static BwppStr bwpp_shape_one_dim(void) {
  BwppStr one = { "1", 1 };
  return one;
}

static int bwpp_shape_rank(const BwppShape *shape) {
  return (int)shape->rank;
}

static uint32_t bwpp_graph_add_value(BwppGraph *g, BwppGraphValue v) {
  if (g->value_count == g->value_capacity) {
    uint32_t new_cap = g->value_capacity == 0 ? 16 : g->value_capacity * 2;
    BwppGraphValue *nv = (BwppGraphValue *)realloc(g->values, new_cap * sizeof(BwppGraphValue));
    if (!nv) {
      return BWPP_GRAPH_NO_VALUE;
    }
    g->values = nv;
    g->value_capacity = new_cap;
  }
  v.id = g->value_count;
  g->values[g->value_count++] = v;
  return v.id;
}

static uint32_t bwpp_graph_add_node(BwppGraph *g, BwppGraphNode n) {
  if (g->node_count == g->node_capacity) {
    uint32_t new_cap = g->node_capacity == 0 ? 16 : g->node_capacity * 2;
    BwppGraphNode *nn = (BwppGraphNode *)realloc(g->nodes, new_cap * sizeof(BwppGraphNode));
    if (!nn) {
      return BWPP_GRAPH_NO_NODE;
    }
    g->nodes = nn;
    g->node_capacity = new_cap;
  }
  n.id = g->node_count;
  g->nodes[g->node_count++] = n;
  return n.id;
}

static void bwpp_graph_add_output(BwppGraph *g, uint32_t value_id) {
  if (g->output_count == g->output_capacity) {
    uint32_t new_cap = g->output_capacity == 0 ? 4 : g->output_capacity * 2;
    uint32_t *no = (uint32_t *)realloc(g->outputs, new_cap * sizeof(uint32_t));
    if (!no) {
      return;
    }
    g->outputs = no;
    g->output_capacity = new_cap;
  }
  g->outputs[g->output_count++] = value_id;
}

static void bwpp_binding_set(BwppGraphBuilder *b, BwppStr name, uint32_t value_id) {
  for (uint32_t i = 0; i < b->binding_count; ++i) {
    if (bwpp_str_eq_str(b->bindings[i].name, name)) {
      b->bindings[i].value_id = value_id;
      return;
    }
  }
  if (b->binding_count == b->binding_capacity) {
    uint32_t new_cap = b->binding_capacity == 0 ? 16 : b->binding_capacity * 2;
    BwppBinding *nb = (BwppBinding *)realloc(b->bindings, new_cap * sizeof(BwppBinding));
    if (!nb) {
      return;
    }
    b->bindings = nb;
    b->binding_capacity = new_cap;
  }
  b->bindings[b->binding_count].name = name;
  b->bindings[b->binding_count].value_id = value_id;
  b->binding_count++;
}

static uint32_t bwpp_binding_get(BwppGraphBuilder *b, BwppStr name) {
  for (uint32_t i = 0; i < b->binding_count; ++i) {
    if (bwpp_str_eq_str(b->bindings[i].name, name)) {
      return b->bindings[i].value_id;
    }
  }
  return BWPP_GRAPH_NO_VALUE;
}

static uint32_t bwpp_graph_get_or_add_input(BwppGraphBuilder *b, BwppStr name) {
  uint32_t existing = bwpp_binding_get(b, name);
  if (existing != BWPP_GRAPH_NO_VALUE) {
    return existing;
  }
  BwppGraphValue v = {0};
  v.name = name;
  v.dtype = BWPP_DTYPE_UNKNOWN;
  v.layout = BWPP_LAYOUT_UNKNOWN;
  v.shape.rank = 0;
  v.producer = BWPP_GRAPH_NO_NODE;
  v.flags = BWPP_GRAPH_VALUE_INPUT;
  uint32_t id = bwpp_graph_add_value(b->graph, v);
  if (id != BWPP_GRAPH_NO_VALUE) {
    bwpp_binding_set(b, name, id);
  }
  return id;
}

static uint32_t bwpp_graph_add_op_node(BwppGraph *g,
                                       BwppGraphOpKind op,
                                       uint32_t *inputs,
                                       uint32_t input_count,
                                       const BwppGraphAttr *attr,
                                       const BwppShape *out_shape,
                                       BwppDType dtype,
                                       BwppLayout layout,
                                       uint32_t flags) {
  BwppGraphValue out = {0};
  out.name = (BwppStr){0};
  out.dtype = dtype;
  out.layout = layout;
  if (out_shape) {
    bwpp_shape_copy(&out.shape, out_shape);
  }
  out.producer = BWPP_GRAPH_NO_NODE;
  out.flags = 0;
  uint32_t out_id = bwpp_graph_add_value(g, out);
  if (out_id == BWPP_GRAPH_NO_VALUE) {
    return BWPP_GRAPH_NO_VALUE;
  }

  BwppGraphNode node = {0};
  node.op = op;
  node.input_count = input_count;
  for (uint32_t i = 0; i < input_count && i < BWPP_GRAPH_MAX_INPUTS; ++i) {
    node.inputs[i] = inputs[i];
  }
  node.output = out_id;
  node.region_id = BWPP_GRAPH_NO_REGION;
  node.flags = flags;
  if (attr) {
    node.attr = *attr;
  }
  uint32_t node_id = bwpp_graph_add_node(g, node);
  if (node_id == BWPP_GRAPH_NO_NODE) {
    return BWPP_GRAPH_NO_VALUE;
  }
  g->values[out_id].producer = node_id;
  return out_id;
}

static uint32_t bwpp_parse_expr(BwppGraphParser *p, BwppGraphBuilder *b);

static uint32_t bwpp_parse_primary(BwppGraphParser *p, BwppGraphBuilder *b) {
  BwppToken tok = bwpp_graph_next(p);
  if (tok.kind == BWPP_TOK_IDENT) {
    BwppStr name = bwpp_tok_str(&tok);
    BwppToken next = bwpp_graph_next(p);
    if (next.kind == BWPP_TOK_SYMBOL && next.length == 1 && next.lexeme[0] == '(') {
      BwppGraphAttr attr = {0};
      uint32_t args[BWPP_GRAPH_MAX_INPUTS] = {0};
      uint32_t argc = 0;
      int expect_more = 1;
      if (bwpp_tok_is(&tok, "reshape") || bwpp_tok_is(&tok, "permute")) {
        uint32_t input = bwpp_parse_expr(p, b);
        args[argc++] = input;
        BwppToken comma = bwpp_graph_next(p);
        if (!(comma.kind == BWPP_TOK_SYMBOL && comma.length == 1 && comma.lexeme[0] == ',')) {
          return BWPP_GRAPH_NO_VALUE;
        }
        if (bwpp_tok_is(&tok, "reshape")) {
          if (!bwpp_parse_shape_list(p, &attr.shape)) {
            return BWPP_GRAPH_NO_VALUE;
          }
        } else {
          BwppShape perm = {0};
          if (!bwpp_parse_shape_list(p, &perm)) {
            return BWPP_GRAPH_NO_VALUE;
          }
          attr.perm_rank = perm.rank;
          for (uint32_t i = 0; i < perm.rank && i < BWPP_GRAPH_MAX_DIMS; ++i) {
            uint32_t axis = 0;
            if (bwpp_str_to_u32(perm.dims[i], &axis)) {
              attr.perm[i] = axis;
            } else {
              attr.perm[i] = i;
            }
          }
        }
        BwppToken close = bwpp_graph_next(p);
        if (!(close.kind == BWPP_TOK_SYMBOL && close.length == 1 && close.lexeme[0] == ')')) {
          return BWPP_GRAPH_NO_VALUE;
        }
        BwppShape out_shape = {0};
        BwppShape in_shape = b->graph->values[input].shape;
        if (bwpp_tok_is(&tok, "reshape")) {
          out_shape = attr.shape;
        } else {
          out_shape.rank = in_shape.rank;
          if (attr.perm_rank == in_shape.rank) {
            for (uint32_t i = 0; i < in_shape.rank && i < BWPP_GRAPH_MAX_DIMS; ++i) {
              uint32_t src = attr.perm[i];
              if (src < in_shape.rank) {
                out_shape.dims[i] = in_shape.dims[src];
              }
            }
          } else {
            bwpp_shape_copy(&out_shape, &in_shape);
          }
        }
        BwppGraphOpKind op = bwpp_tok_is(&tok, "reshape") ? BWPP_GOP_RESHAPE : BWPP_GOP_PERMUTE;
        return bwpp_graph_add_op_node(b->graph, op, args, argc, &attr, &out_shape,
                                      b->graph->values[input].dtype,
                                      b->graph->values[input].layout,
                                      0);
      }

      if (bwpp_tok_is(&tok, "transpose")) {
        uint32_t input = bwpp_parse_expr(p, b);
        args[argc++] = input;
        BwppToken close = bwpp_graph_next(p);
        if (!(close.kind == BWPP_TOK_SYMBOL && close.length == 1 && close.lexeme[0] == ')')) {
          return BWPP_GRAPH_NO_VALUE;
        }
        BwppShape out_shape = b->graph->values[input].shape;
        if (out_shape.rank == 2) {
          BwppStr tmp = out_shape.dims[0];
          out_shape.dims[0] = out_shape.dims[1];
          out_shape.dims[1] = tmp;
        }
        return bwpp_graph_add_op_node(b->graph, BWPP_GOP_TRANSPOSE, args, argc, NULL, &out_shape,
                                      b->graph->values[input].dtype,
                                      b->graph->values[input].layout,
                                      0);
      }

      if (bwpp_tok_is(&tok, "softmax") || bwpp_tok_is(&tok, "silu")) {
        uint32_t input = bwpp_parse_expr(p, b);
        args[argc++] = input;
        BwppToken maybe = bwpp_graph_next(p);
        if (maybe.kind == BWPP_TOK_SYMBOL && maybe.length == 1 && maybe.lexeme[0] == ',') {
          BwppToken axis_tok = bwpp_graph_next(p);
          if (axis_tok.kind == BWPP_TOK_NUMBER) {
            BwppStr axis_str = bwpp_tok_str(&axis_tok);
            uint32_t axis = 0;
            if (bwpp_str_to_u32(axis_str, &axis)) {
              attr.has_axis = 1;
              attr.axis = (int)axis;
            }
          }
          BwppToken close2 = bwpp_graph_next(p);
          if (!(close2.kind == BWPP_TOK_SYMBOL && close2.length == 1 && close2.lexeme[0] == ')')) {
            return BWPP_GRAPH_NO_VALUE;
          }
        } else {
          if (!(maybe.kind == BWPP_TOK_SYMBOL && maybe.length == 1 && maybe.lexeme[0] == ')')) {
            return BWPP_GRAPH_NO_VALUE;
          }
        }
        BwppGraphOpKind op = bwpp_tok_is(&tok, "softmax") ? BWPP_GOP_SOFTMAX : BWPP_GOP_SILU;
        return bwpp_graph_add_op_node(b->graph, op, args, argc, &attr,
                                      &b->graph->values[input].shape,
                                      b->graph->values[input].dtype,
                                      b->graph->values[input].layout,
                                      0);
      }

      if (bwpp_tok_is(&tok, "rmsnorm")) {
        uint32_t input = bwpp_parse_expr(p, b);
        args[argc++] = input;
        BwppToken comma = bwpp_graph_next(p);
        if (!(comma.kind == BWPP_TOK_SYMBOL && comma.length == 1 && comma.lexeme[0] == ',')) {
          return BWPP_GRAPH_NO_VALUE;
        }
        uint32_t gamma = bwpp_parse_expr(p, b);
        args[argc++] = gamma;
        BwppToken next_tok = bwpp_graph_next(p);
        if (next_tok.kind == BWPP_TOK_SYMBOL && next_tok.length == 1 && next_tok.lexeme[0] == ',') {
          BwppToken peek = bwpp_graph_next(p);
          if (peek.kind == BWPP_TOK_NUMBER) {
            float eps = 1e-5f;
            float parsed = 0.0f;
            if (bwpp_str_to_f32(bwpp_tok_str(&peek), &parsed)) {
              eps = parsed;
            }
            BwppToken close = bwpp_graph_next(p);
            attr.has_epsilon = 1;
            attr.epsilon = eps;
            if (!(close.kind == BWPP_TOK_SYMBOL && close.length == 1 && close.lexeme[0] == ')')) {
              return BWPP_GRAPH_NO_VALUE;
            }
          } else {
            bwpp_graph_unread(p, peek);
            uint32_t beta = bwpp_parse_expr(p, b);
            args[argc++] = beta;
            BwppToken next2 = bwpp_graph_next(p);
            if (next2.kind == BWPP_TOK_SYMBOL && next2.length == 1 && next2.lexeme[0] == ',') {
              BwppToken eps_tok = bwpp_graph_next(p);
              float eps = 1e-5f;
              float parsed = 0.0f;
              if (eps_tok.kind == BWPP_TOK_NUMBER &&
                  bwpp_str_to_f32(bwpp_tok_str(&eps_tok), &parsed)) {
                eps = parsed;
              }
              BwppToken close = bwpp_graph_next(p);
              attr.has_epsilon = 1;
              attr.epsilon = eps;
              if (!(close.kind == BWPP_TOK_SYMBOL && close.length == 1 && close.lexeme[0] == ')')) {
                return BWPP_GRAPH_NO_VALUE;
              }
            } else if (next2.kind == BWPP_TOK_SYMBOL && next2.length == 1 && next2.lexeme[0] == ')') {
              attr.has_epsilon = 1;
              attr.epsilon = 1e-5f;
            } else {
              return BWPP_GRAPH_NO_VALUE;
            }
          }
        } else if (next_tok.kind == BWPP_TOK_SYMBOL && next_tok.length == 1 && next_tok.lexeme[0] == ')') {
          attr.has_epsilon = 1;
          attr.epsilon = 1e-5f;
        } else {
          return BWPP_GRAPH_NO_VALUE;
        }
        return bwpp_graph_add_op_node(b->graph, BWPP_GOP_RMSNORM, args, argc, &attr,
                                      &b->graph->values[input].shape,
                                      b->graph->values[input].dtype,
                                      b->graph->values[input].layout,
                                      0);
      }

      if (bwpp_tok_is(&tok, "reduce_sum") || bwpp_tok_is(&tok, "reduce_max")) {
        uint32_t input = bwpp_parse_expr(p, b);
        args[argc++] = input;
        BwppToken maybe = bwpp_graph_next(p);
        if (maybe.kind == BWPP_TOK_SYMBOL && maybe.length == 1 && maybe.lexeme[0] == ',') {
          BwppToken axis_tok = bwpp_graph_next(p);
          if (axis_tok.kind == BWPP_TOK_NUMBER) {
            BwppStr axis_str = bwpp_tok_str(&axis_tok);
            uint32_t axis = 0;
            if (bwpp_str_to_u32(axis_str, &axis)) {
              attr.has_axis = 1;
              attr.axis = (int)axis;
            }
          }
          BwppToken close = bwpp_graph_next(p);
          if (!(close.kind == BWPP_TOK_SYMBOL && close.length == 1 && close.lexeme[0] == ')')) {
            return BWPP_GRAPH_NO_VALUE;
          }
        } else {
          if (!(maybe.kind == BWPP_TOK_SYMBOL && maybe.length == 1 && maybe.lexeme[0] == ')')) {
            return BWPP_GRAPH_NO_VALUE;
          }
        }
        BwppGraphOpKind op = bwpp_tok_is(&tok, "reduce_sum") ? BWPP_GOP_REDUCE_SUM : BWPP_GOP_REDUCE_MAX;
        BwppShape out_shape = b->graph->values[input].shape;
        if (attr.has_axis && attr.axis >= 0 && (uint32_t)attr.axis < out_shape.rank) {
          out_shape.dims[attr.axis] = bwpp_shape_one_dim();
        }
        return bwpp_graph_add_op_node(b->graph, op, args, argc, &attr, &out_shape,
                                      b->graph->values[input].dtype,
                                      b->graph->values[input].layout,
                                      0);
      }

      if (bwpp_tok_is(&tok, "matmul") || bwpp_tok_is(&tok, "batch_matmul") ||
          bwpp_tok_is(&tok, "add") || bwpp_tok_is(&tok, "sub") ||
          bwpp_tok_is(&tok, "mul") || bwpp_tok_is(&tok, "div")) {
        while (expect_more) {
          BwppToken peek = bwpp_graph_next(p);
          if (peek.kind == BWPP_TOK_SYMBOL && peek.length == 1 && peek.lexeme[0] == ')') {
            break;
          }
          bwpp_graph_unread(p, peek);
          uint32_t arg = bwpp_parse_expr(p, b);
          if (argc < BWPP_GRAPH_MAX_INPUTS) {
            args[argc++] = arg;
          }
          BwppToken sep = bwpp_graph_next(p);
          if (sep.kind == BWPP_TOK_SYMBOL && sep.length == 1 && sep.lexeme[0] == ',') {
            continue;
          }
          if (sep.kind == BWPP_TOK_SYMBOL && sep.length == 1 && sep.lexeme[0] == ')') {
            break;
          }
          return BWPP_GRAPH_NO_VALUE;
        }

        BwppGraphOpKind op = BWPP_GOP_ADD;
        if (bwpp_tok_is(&tok, "matmul")) {
          op = BWPP_GOP_MATMUL;
        } else if (bwpp_tok_is(&tok, "batch_matmul")) {
          op = BWPP_GOP_BATCH_MATMUL;
        } else if (bwpp_tok_is(&tok, "add")) {
          op = BWPP_GOP_ADD;
        } else if (bwpp_tok_is(&tok, "sub")) {
          op = BWPP_GOP_SUB;
        } else if (bwpp_tok_is(&tok, "mul")) {
          op = BWPP_GOP_MUL;
        } else if (bwpp_tok_is(&tok, "div")) {
          op = BWPP_GOP_DIV;
        }

        BwppShape out_shape = {0};
        BwppDType dtype = BWPP_DTYPE_UNKNOWN;
        BwppLayout layout = BWPP_LAYOUT_UNKNOWN;
        if (argc >= 1) {
          dtype = b->graph->values[args[0]].dtype;
          layout = b->graph->values[args[0]].layout;
          bwpp_shape_copy(&out_shape, &b->graph->values[args[0]].shape);
        }
        if (op == BWPP_GOP_MATMUL && argc >= 2) {
          BwppShape a = b->graph->values[args[0]].shape;
          BwppShape bshape = b->graph->values[args[1]].shape;
          if (a.rank == 2 && bshape.rank == 2) {
            out_shape.rank = 2;
            out_shape.dims[0] = a.dims[0];
            out_shape.dims[1] = bshape.dims[1];
          }
        } else if (op == BWPP_GOP_BATCH_MATMUL && argc >= 2) {
          bwpp_shape_copy(&out_shape, &b->graph->values[args[0]].shape);
        } else if ((op == BWPP_GOP_ADD || op == BWPP_GOP_SUB || op == BWPP_GOP_MUL || op == BWPP_GOP_DIV) &&
                   argc >= 2) {
          BwppShape a = b->graph->values[args[0]].shape;
          BwppShape bshape = b->graph->values[args[1]].shape;
          out_shape = bwpp_shape_broadcast(&a, &bshape);
        }

        uint32_t flags = 0;
        if (op == BWPP_GOP_ADD) {
          for (uint32_t i = 0; i < argc; ++i) {
            BwppStr nm = b->graph->values[args[i]].name;
            if (nm.ptr && bwpp_str_eq(nm, "bias")) {
              flags |= BWPP_GRAPH_OPF_HAS_BIAS;
            }
          }
        }
        return bwpp_graph_add_op_node(b->graph, op, args, argc, &attr, &out_shape, dtype, layout, flags);
      }

      bwpp_graph_unread(p, next);
      return bwpp_graph_get_or_add_input(b, name);
    }
    bwpp_graph_unread(p, next);
    return bwpp_graph_get_or_add_input(b, name);
  }
  if (tok.kind == BWPP_TOK_NUMBER) {
    BwppGraphValue v = {0};
    v.dtype = BWPP_DTYPE_F32;
    v.layout = BWPP_LAYOUT_UNKNOWN;
    v.shape.rank = 0;
    v.producer = BWPP_GRAPH_NO_NODE;
    v.flags = BWPP_GRAPH_VALUE_CONST;
    return bwpp_graph_add_value(b->graph, v);
  }
  if (tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '(') {
    uint32_t val = bwpp_parse_expr(p, b);
    BwppToken close = bwpp_graph_next(p);
    if (!(close.kind == BWPP_TOK_SYMBOL && close.length == 1 && close.lexeme[0] == ')')) {
      return BWPP_GRAPH_NO_VALUE;
    }
    return val;
  }
  return BWPP_GRAPH_NO_VALUE;
}

static uint32_t bwpp_parse_expr(BwppGraphParser *p, BwppGraphBuilder *b) {
  uint32_t lhs = bwpp_parse_primary(p, b);
  if (lhs == BWPP_GRAPH_NO_VALUE) {
    return lhs;
  }
  for (;;) {
    BwppToken tok = bwpp_graph_next(p);
    if (tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '@') {
      uint32_t rhs = bwpp_parse_primary(p, b);
      uint32_t inputs[2] = { lhs, rhs };
      BwppShape out_shape = {0};
      BwppShape a = b->graph->values[lhs].shape;
      BwppShape bshape = b->graph->values[rhs].shape;
      if (a.rank == 2 && bshape.rank == 2) {
        out_shape.rank = 2;
        out_shape.dims[0] = a.dims[0];
        out_shape.dims[1] = bshape.dims[1];
      }
      lhs = bwpp_graph_add_op_node(b->graph, BWPP_GOP_MATMUL, inputs, 2, NULL, &out_shape,
                                   b->graph->values[lhs].dtype,
                                   b->graph->values[lhs].layout,
                                   0);
      continue;
    }
    bwpp_graph_unread(p, tok);
    break;
  }
  return lhs;
}

static int bwpp_parse_params(BwppGraphParser *p, BwppGraphBuilder *b) {
  BwppToken tok = bwpp_graph_next(p);
  if (!(tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '(')) {
    return 0;
  }
  for (;;) {
    BwppToken name = bwpp_graph_next(p);
    if (name.kind == BWPP_TOK_SYMBOL && name.length == 1 && name.lexeme[0] == ')') {
      break;
    }
    if (name.kind != BWPP_TOK_IDENT) {
      return 0;
    }
    BwppToken colon = bwpp_graph_next(p);
    if (!(colon.kind == BWPP_TOK_SYMBOL && colon.length == 1 && colon.lexeme[0] == ':')) {
      return 0;
    }
    BwppToken tensor = bwpp_graph_next(p);
    if (!bwpp_tok_is(&tensor, "tensor")) {
      return 0;
    }
    BwppToken lt = bwpp_graph_next(p);
    if (!(lt.kind == BWPP_TOK_SYMBOL && lt.length == 1 && lt.lexeme[0] == '<')) {
      return 0;
    }
    BwppToken dtype_tok = bwpp_graph_next(p);
    if (dtype_tok.kind != BWPP_TOK_IDENT) {
      return 0;
    }
    BwppToken comma = bwpp_graph_next(p);
    if (!(comma.kind == BWPP_TOK_SYMBOL && comma.length == 1 && comma.lexeme[0] == ',')) {
      return 0;
    }
    BwppShape shape = {0};
    if (!bwpp_parse_shape_list(p, &shape)) {
      return 0;
    }
    BwppLayout layout = BWPP_LAYOUT_UNKNOWN;
    BwppToken next = bwpp_graph_next(p);
    if (next.kind == BWPP_TOK_SYMBOL && next.length == 1 && next.lexeme[0] == ',') {
      BwppToken layout_tok = bwpp_graph_next(p);
      if (layout_tok.kind == BWPP_TOK_IDENT) {
        layout = bwpp_parse_layout(bwpp_tok_str(&layout_tok));
      }
      BwppToken gt = bwpp_graph_next(p);
      if (!(gt.kind == BWPP_TOK_SYMBOL && gt.length == 1 && gt.lexeme[0] == '>')) {
        return 0;
      }
    } else if (next.kind == BWPP_TOK_SYMBOL && next.length == 1 && next.lexeme[0] == '>') {
      /* no layout */
    } else {
      return 0;
    }

    BwppGraphValue v = {0};
    v.name = bwpp_tok_str(&name);
    v.dtype = bwpp_parse_dtype(bwpp_tok_str(&dtype_tok));
    v.layout = layout;
    bwpp_shape_copy(&v.shape, &shape);
    v.producer = BWPP_GRAPH_NO_NODE;
    v.flags = BWPP_GRAPH_VALUE_INPUT;
    uint32_t id = bwpp_graph_add_value(b->graph, v);
    if (id != BWPP_GRAPH_NO_VALUE) {
      bwpp_binding_set(b, v.name, id);
    }

    BwppToken sep = bwpp_graph_next(p);
    if (sep.kind == BWPP_TOK_SYMBOL && sep.length == 1 && sep.lexeme[0] == ')') {
      break;
    }
    if (!(sep.kind == BWPP_TOK_SYMBOL && sep.length == 1 && sep.lexeme[0] == ',')) {
      return 0;
    }
  }
  return 1;
}

BwppGraph *bwpp_graph_build(const BwppAstModule *module) {
  if (!module || !module->source) {
    return NULL;
  }
  BwppGraph *graph = (BwppGraph *)calloc(1, sizeof(BwppGraph));
  if (!graph) {
    return NULL;
  }
  BwppGraphParser parser;
  bwpp_lexer_init(&parser.lx, module->source, module->length);
  parser.has_lookahead = 0;

  BwppGraphBuilder builder = {0};
  builder.graph = graph;

  int brace_depth = 0;
  int pending_reversible = 0;
  int inside_fn = 0;
  int built_fn = 0;
  uint32_t current_region = BWPP_GRAPH_NO_REGION;
  int reversible_brace_depth = -1;

  for (;;) {
    BwppToken tok = bwpp_graph_next(&parser);
    if (tok.kind == BWPP_TOK_EOF) {
      break;
    }
    if (tok.kind == BWPP_TOK_SYMBOL && tok.length == 1) {
      char ch = tok.lexeme[0];
      if (ch == '@') {
        BwppToken next = bwpp_graph_next(&parser);
        if (bwpp_tok_is(&next, "reversible")) {
          pending_reversible = 1;
        }
        continue;
      }
      if (ch == '{') {
        brace_depth++;
        if (inside_fn && pending_reversible && current_region == BWPP_GRAPH_NO_REGION) {
          BwppGraphRegion reg = {0};
          reg.id = graph->region_count;
          reg.kind = BWPP_REGION_REVERSIBLE;
          reg.policy = BWPP_POLICY_AUTO;
          if (graph->region_count == graph->region_capacity) {
            uint32_t new_cap = graph->region_capacity == 0 ? 4 : graph->region_capacity * 2;
            BwppGraphRegion *nr = (BwppGraphRegion *)realloc(graph->regions, new_cap * sizeof(BwppGraphRegion));
            if (!nr) {
              break;
            }
            graph->regions = nr;
            graph->region_capacity = new_cap;
          }
          graph->regions[graph->region_count++] = reg;
          current_region = reg.id;
          reversible_brace_depth = brace_depth;
          pending_reversible = 0;
        }
        continue;
      }
      if (ch == '}') {
        if (current_region != BWPP_GRAPH_NO_REGION && reversible_brace_depth == brace_depth) {
          current_region = BWPP_GRAPH_NO_REGION;
          reversible_brace_depth = -1;
        }
        brace_depth--;
        if (inside_fn && brace_depth == 0) {
          inside_fn = 0;
          built_fn = 1;
          break;
        }
        continue;
      }
    }

    if (tok.kind == BWPP_TOK_IDENT && bwpp_tok_is(&tok, "fn")) {
      if (built_fn) {
        break;
      }
      BwppToken name = bwpp_graph_next(&parser);
      (void)name;
      inside_fn = 1;
      if (!bwpp_parse_params(&parser, &builder)) {
        break;
      }
      continue;
    }

    if (!inside_fn || brace_depth <= 0) {
      continue;
    }

    if (tok.kind == BWPP_TOK_IDENT && bwpp_tok_is(&tok, "let")) {
      BwppToken name = bwpp_graph_next(&parser);
      if (name.kind != BWPP_TOK_IDENT) {
        continue;
      }
      BwppToken eq = bwpp_graph_next(&parser);
      if (!(eq.kind == BWPP_TOK_SYMBOL && eq.length == 1 && eq.lexeme[0] == '=')) {
        continue;
      }
      uint32_t val = bwpp_parse_expr(&parser, &builder);
      if (val != BWPP_GRAPH_NO_VALUE) {
        graph->values[val].name = bwpp_tok_str(&name);
        bwpp_binding_set(&builder, graph->values[val].name, val);
        if (current_region != BWPP_GRAPH_NO_REGION && graph->values[val].producer != BWPP_GRAPH_NO_NODE) {
          graph->nodes[graph->values[val].producer].region_id = current_region;
        }
      }
      continue;
    }

    if (tok.kind == BWPP_TOK_IDENT && bwpp_tok_is(&tok, "return")) {
      uint32_t val = bwpp_parse_expr(&parser, &builder);
      if (val != BWPP_GRAPH_NO_VALUE) {
        graph->values[val].flags |= BWPP_GRAPH_VALUE_OUTPUT;
        bwpp_graph_add_output(graph, val);
      }
      continue;
    }
  }

  free(builder.bindings);
  return graph;
}

static const char *bwpp_op_name(BwppGraphOpKind op) {
  switch (op) {
    case BWPP_GOP_MATMUL: return "matmul";
    case BWPP_GOP_BATCH_MATMUL: return "batch_matmul";
    case BWPP_GOP_TRANSPOSE: return "transpose";
    case BWPP_GOP_PERMUTE: return "permute";
    case BWPP_GOP_RESHAPE: return "reshape";
    case BWPP_GOP_BROADCAST: return "broadcast";
    case BWPP_GOP_ADD: return "add";
    case BWPP_GOP_SUB: return "sub";
    case BWPP_GOP_MUL: return "mul";
    case BWPP_GOP_DIV: return "div";
    case BWPP_GOP_REDUCE_SUM: return "reduce_sum";
    case BWPP_GOP_REDUCE_MAX: return "reduce_max";
    case BWPP_GOP_REDUCE_MAX_MASK: return "reduce_max_mask";
    case BWPP_GOP_REDUCE_MAX_GRAD: return "reduce_max_grad";
    case BWPP_GOP_SOFTMAX: return "softmax";
    case BWPP_GOP_RMSNORM: return "rmsnorm";
    case BWPP_GOP_SILU: return "silu";
    case BWPP_GOP_SILU_GRAD: return "silu_grad";
    case BWPP_GOP_SOFTMAX_GRAD: return "softmax_grad";
    case BWPP_GOP_RMSNORM_GRAD: return "rmsnorm_grad";
    default: return "unknown";
  }
}

static const char *bwpp_dtype_name(BwppDType dt) {
  switch (dt) {
    case BWPP_DTYPE_F16: return "f16";
    case BWPP_DTYPE_BF16: return "bf16";
    case BWPP_DTYPE_F32: return "f32";
    default: return "unknown";
  }
}

static void bwpp_print_shape(FILE *out, const BwppShape *shape) {
  if (!out) {
    return;
  }
  fprintf(out, "[");
  for (uint32_t i = 0; i < shape->rank; ++i) {
    if (i > 0) {
      fprintf(out, ",");
    }
    if (shape->dims[i].ptr) {
      fprintf(out, "%.*s", (int)shape->dims[i].len, shape->dims[i].ptr);
    }
  }
  fprintf(out, "]");
}

void bwpp_graph_dump(const BwppGraph *graph, FILE *out) {
  if (!graph || !out) {
    return;
  }
  fprintf(out, "graph values=%u nodes=%u outputs=%u\n",
          graph->value_count, graph->node_count, graph->output_count);
  for (uint32_t i = 0; i < graph->value_count; ++i) {
    const BwppGraphValue *v = &graph->values[i];
    fprintf(out, "  v%u ", v->id);
    if (v->name.ptr) {
      fprintf(out, "%.*s ", (int)v->name.len, v->name.ptr);
    }
    fprintf(out, "%s ", bwpp_dtype_name(v->dtype));
    bwpp_print_shape(out, &v->shape);
    fprintf(out, " flags=0x%x\n", v->flags);
  }
  for (uint32_t i = 0; i < graph->node_count; ++i) {
    const BwppGraphNode *n = &graph->nodes[i];
    fprintf(out, "  n%u %s (", n->id, bwpp_op_name(n->op));
    for (uint32_t j = 0; j < n->input_count; ++j) {
      if (j) {
        fprintf(out, ",");
      }
      fprintf(out, "v%u", n->inputs[j]);
    }
    fprintf(out, ") -> v%u\n", n->output);
  }
}

void bwpp_graph_dump_dot(const BwppGraph *graph, FILE *out) {
  if (!graph || !out) {
    return;
  }
  fprintf(out, "digraph bwpp {\n");
  fprintf(out, "  rankdir=LR;\n");
  fprintf(out, "  node [fontname=\"Menlo\"];\n");
  for (uint32_t i = 0; i < graph->value_count; ++i) {
    const BwppGraphValue *v = &graph->values[i];
    fprintf(out, "  v%u [shape=ellipse", v->id);
    if (v->flags & BWPP_GRAPH_VALUE_INPUT) {
      fprintf(out, ", style=filled, fillcolor=\"#E7F0FF\"");
    } else if (v->flags & BWPP_GRAPH_VALUE_OUTPUT) {
      fprintf(out, ", style=filled, fillcolor=\"#E8FFE7\"");
    }
    fprintf(out, ", label=\"");
    if (v->name.ptr) {
      fprintf(out, "%.*s\\n", (int)v->name.len, v->name.ptr);
    }
    fprintf(out, "%s ", bwpp_dtype_name(v->dtype));
    bwpp_print_shape(out, &v->shape);
    fprintf(out, "\"];\n");
  }
  for (uint32_t i = 0; i < graph->node_count; ++i) {
    const BwppGraphNode *n = &graph->nodes[i];
    fprintf(out, "  n%u [shape=box, label=\"%s\"];\n", n->id, bwpp_op_name(n->op));
    for (uint32_t j = 0; j < n->input_count; ++j) {
      fprintf(out, "  v%u -> n%u;\n", n->inputs[j], n->id);
    }
    fprintf(out, "  n%u -> v%u;\n", n->id, n->output);
  }
  fprintf(out, "}\n");
}

static uint32_t bwpp_graph_clone_value(BwppGraph *dst,
                                       const BwppGraph *src,
                                       uint32_t value_id,
                                       uint32_t flags) {
  const BwppGraphValue *v = &src->values[value_id];
  BwppGraphValue out = {0};
  out.name = v->name;
  out.dtype = v->dtype;
  out.layout = v->layout;
  bwpp_shape_copy(&out.shape, &v->shape);
  out.producer = BWPP_GRAPH_NO_NODE;
  out.flags = flags;
  return bwpp_graph_add_value(dst, out);
}

static uint32_t bwpp_graph_import_activation(BwppGraph *dst,
                                             const BwppGraph *src,
                                             uint32_t *act_map,
                                             uint32_t value_id) {
  if (act_map[value_id] != BWPP_GRAPH_NO_VALUE) {
    return act_map[value_id];
  }
  uint32_t id = bwpp_graph_clone_value(dst, src, value_id, BWPP_GRAPH_VALUE_INPUT);
  act_map[value_id] = id;
  return id;
}

static uint32_t bwpp_graph_const_scalar(BwppGraph *g, const char *name, BwppDType dtype) {
  BwppGraphValue v = {0};
  v.name.ptr = name;
  v.name.len = strlen(name);
  v.dtype = dtype;
  v.layout = BWPP_LAYOUT_UNKNOWN;
  v.shape.rank = 0;
  v.producer = BWPP_GRAPH_NO_NODE;
  v.flags = BWPP_GRAPH_VALUE_CONST;
  return bwpp_graph_add_value(g, v);
}

static uint32_t bwpp_graph_negate(BwppGraph *g, uint32_t val) {
  uint32_t c = bwpp_graph_const_scalar(g, "-1", g->values[val].dtype);
  uint32_t inputs[2] = { val, c };
  BwppShape out_shape = g->values[val].shape;
  BwppGraphAttr attr = {0};
  return bwpp_graph_add_op_node(g, BWPP_GOP_MUL, inputs, 2, &attr, &out_shape,
                                g->values[val].dtype,
                                g->values[val].layout,
                                0);
}

static uint32_t bwpp_graph_reduce_to_shape(BwppGraph *g,
                                           uint32_t val,
                                           const BwppShape *target_shape) {
  BwppShape cur = g->values[val].shape;
  if (bwpp_shape_equal(&cur, target_shape)) {
    return val;
  }
  uint32_t cur_rank = cur.rank;
  uint32_t tgt_rank = target_shape->rank;
  for (uint32_t axis = 0; axis < cur_rank; ++axis) {
    int tgt_idx = (int)axis - (int)(cur_rank - tgt_rank);
    int reduce = 0;
    if (tgt_idx < 0) {
      reduce = 1;
    } else {
      BwppStr tgt_dim = target_shape->dims[tgt_idx];
      BwppStr cur_dim = cur.dims[axis];
      if (bwpp_str_eq(tgt_dim, "1") && !bwpp_str_eq(cur_dim, "1")) {
        reduce = 1;
      }
    }
    if (reduce) {
      BwppGraphAttr attr = {0};
      attr.has_axis = 1;
      attr.axis = (int)axis;
      BwppShape out_shape = cur;
      out_shape.dims[axis] = bwpp_shape_one_dim();
      uint32_t inputs[1] = { val };
      val = bwpp_graph_add_op_node(g, BWPP_GOP_REDUCE_SUM, inputs, 1, &attr, &out_shape,
                                   g->values[val].dtype,
                                   g->values[val].layout,
                                   0);
      cur = out_shape;
    }
  }
  if (cur.rank != target_shape->rank && target_shape->rank > 0) {
    BwppGraphAttr attr = {0};
    attr.shape = *target_shape;
    uint32_t inputs[1] = { val };
    val = bwpp_graph_add_op_node(g, BWPP_GOP_RESHAPE, inputs, 1, &attr, target_shape,
                                 g->values[val].dtype,
                                 g->values[val].layout,
                                 0);
  }
  return val;
}

static uint32_t bwpp_graph_broadcast_to_shape(BwppGraph *g,
                                              uint32_t val,
                                              const BwppShape *target_shape) {
  if (bwpp_shape_equal(&g->values[val].shape, target_shape)) {
    return val;
  }
  BwppGraphAttr attr = {0};
  attr.shape = *target_shape;
  uint32_t inputs[1] = { val };
  return bwpp_graph_add_op_node(g, BWPP_GOP_BROADCAST, inputs, 1, &attr, target_shape,
                                g->values[val].dtype,
                                g->values[val].layout,
                                0);
}

static uint32_t bwpp_graph_accum_grad(BwppGraph *g, uint32_t existing, uint32_t add_val) {
  if (existing == BWPP_GRAPH_NO_VALUE) {
    return add_val;
  }
  uint32_t inputs[2] = { existing, add_val };
  BwppShape out_shape = g->values[existing].shape;
  BwppGraphAttr attr = {0};
  return bwpp_graph_add_op_node(g, BWPP_GOP_ADD, inputs, 2, &attr, &out_shape,
                                g->values[existing].dtype,
                                g->values[existing].layout,
                                0);
}

BwppGraph *bwpp_graph_autodiff(const BwppGraph *graph) {
  if (!graph) {
    return NULL;
  }
  BwppGraph *grad = (BwppGraph *)calloc(1, sizeof(BwppGraph));
  if (!grad) {
    return NULL;
  }

  uint32_t *act_map = (uint32_t *)malloc(sizeof(uint32_t) * graph->value_count);
  uint32_t *grad_map = (uint32_t *)malloc(sizeof(uint32_t) * graph->value_count);
  if (!act_map || !grad_map) {
    free(act_map);
    free(grad_map);
    bwpp_graph_destroy(grad);
    return NULL;
  }
  for (uint32_t i = 0; i < graph->value_count; ++i) {
    act_map[i] = BWPP_GRAPH_NO_VALUE;
    grad_map[i] = BWPP_GRAPH_NO_VALUE;
  }

  for (uint32_t i = 0; i < graph->value_count; ++i) {
    if (graph->values[i].flags & BWPP_GRAPH_VALUE_INPUT) {
      act_map[i] = bwpp_graph_clone_value(grad, graph, i, BWPP_GRAPH_VALUE_INPUT);
    }
  }

  for (uint32_t i = 0; i < graph->output_count; ++i) {
    uint32_t out_id = graph->outputs[i];
    uint32_t grad_seed = bwpp_graph_clone_value(grad, graph, out_id, BWPP_GRAPH_VALUE_INPUT);
    if (grad_seed != BWPP_GRAPH_NO_VALUE) {
      grad_map[out_id] = grad_seed;
    }
  }

  for (int32_t i = (int32_t)graph->node_count - 1; i >= 0; --i) {
    const BwppGraphNode *n = &graph->nodes[i];
    uint32_t out_id = n->output;
    uint32_t dY = grad_map[out_id];
    if (dY == BWPP_GRAPH_NO_VALUE) {
      continue;
    }

    if (n->op == BWPP_GOP_MATMUL && n->input_count >= 2) {
      uint32_t a = n->inputs[0];
      uint32_t b = n->inputs[1];
      uint32_t actA = bwpp_graph_import_activation(grad, graph, act_map, a);
      uint32_t actB = bwpp_graph_import_activation(grad, graph, act_map, b);

      uint32_t tB_inputs[1] = { actB };
      BwppShape tB_shape = grad->values[actB].shape;
      if (tB_shape.rank == 2) {
        BwppStr tmp = tB_shape.dims[0];
        tB_shape.dims[0] = tB_shape.dims[1];
        tB_shape.dims[1] = tmp;
      }
      uint32_t tB = bwpp_graph_add_op_node(grad, BWPP_GOP_TRANSPOSE, tB_inputs, 1, NULL, &tB_shape,
                                           grad->values[actB].dtype,
                                           grad->values[actB].layout,
                                           0);
      uint32_t dA_inputs[2] = { dY, tB };
      BwppShape dA_shape = {0};
      if (grad->values[dY].shape.rank == 2 && tB_shape.rank == 2) {
        dA_shape.rank = 2;
        dA_shape.dims[0] = grad->values[dY].shape.dims[0];
        dA_shape.dims[1] = tB_shape.dims[1];
      }
      uint32_t dA = bwpp_graph_add_op_node(grad, BWPP_GOP_MATMUL, dA_inputs, 2, NULL, &dA_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      grad_map[a] = bwpp_graph_accum_grad(grad, grad_map[a], dA);

      uint32_t tA_inputs[1] = { actA };
      BwppShape tA_shape = grad->values[actA].shape;
      if (tA_shape.rank == 2) {
        BwppStr tmp = tA_shape.dims[0];
        tA_shape.dims[0] = tA_shape.dims[1];
        tA_shape.dims[1] = tmp;
      }
      uint32_t tA = bwpp_graph_add_op_node(grad, BWPP_GOP_TRANSPOSE, tA_inputs, 1, NULL, &tA_shape,
                                           grad->values[actA].dtype,
                                           grad->values[actA].layout,
                                           0);
      uint32_t dB_inputs[2] = { tA, dY };
      BwppShape dB_shape = {0};
      if (tA_shape.rank == 2 && grad->values[dY].shape.rank == 2) {
        dB_shape.rank = 2;
        dB_shape.dims[0] = tA_shape.dims[0];
        dB_shape.dims[1] = grad->values[dY].shape.dims[1];
      }
      uint32_t dB = bwpp_graph_add_op_node(grad, BWPP_GOP_MATMUL, dB_inputs, 2, NULL, &dB_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      grad_map[b] = bwpp_graph_accum_grad(grad, grad_map[b], dB);
      continue;
    }

    if ((n->op == BWPP_GOP_ADD || n->op == BWPP_GOP_SUB) && n->input_count >= 2) {
      BwppShape shape_a = graph->values[n->inputs[0]].shape;
      BwppShape shape_b = graph->values[n->inputs[1]].shape;
      uint32_t dA = bwpp_graph_reduce_to_shape(grad, dY, &shape_a);
      uint32_t dB = bwpp_graph_reduce_to_shape(grad, dY, &shape_b);
      if (n->op == BWPP_GOP_SUB) {
        dB = bwpp_graph_negate(grad, dB);
      }
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dA);
      grad_map[n->inputs[1]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[1]], dB);
      continue;
    }

    if (n->op == BWPP_GOP_MUL && n->input_count >= 2) {
      uint32_t actA = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[0]);
      uint32_t actB = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[1]);
      uint32_t inputsA[2] = { dY, actB };
      uint32_t inputsB[2] = { dY, actA };
      BwppShape out_shape = grad->values[dY].shape;
      BwppGraphAttr attr = {0};
      uint32_t dA = bwpp_graph_add_op_node(grad, BWPP_GOP_MUL, inputsA, 2, &attr, &out_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      uint32_t dB = bwpp_graph_add_op_node(grad, BWPP_GOP_MUL, inputsB, 2, &attr, &out_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      dA = bwpp_graph_reduce_to_shape(grad, dA, &graph->values[n->inputs[0]].shape);
      dB = bwpp_graph_reduce_to_shape(grad, dB, &graph->values[n->inputs[1]].shape);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dA);
      grad_map[n->inputs[1]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[1]], dB);
      continue;
    }

    if (n->op == BWPP_GOP_DIV && n->input_count >= 2) {
      uint32_t actA = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[0]);
      uint32_t actB = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[1]);
      BwppGraphAttr attr = {0};
      uint32_t dA_inputs[2] = { dY, actB };
      uint32_t dA = bwpp_graph_add_op_node(grad, BWPP_GOP_DIV, dA_inputs, 2, &attr,
                                           &grad->values[dY].shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      uint32_t b2_inputs[2] = { actB, actB };
      uint32_t b2 = bwpp_graph_add_op_node(grad, BWPP_GOP_MUL, b2_inputs, 2, &attr,
                                           &grad->values[actB].shape,
                                           grad->values[actB].dtype,
                                           grad->values[actB].layout,
                                           0);
      uint32_t num_inputs[2] = { dY, actA };
      uint32_t num = bwpp_graph_add_op_node(grad, BWPP_GOP_MUL, num_inputs, 2, &attr,
                                            &grad->values[dY].shape,
                                            grad->values[dY].dtype,
                                            grad->values[dY].layout,
                                            0);
      uint32_t dB_inputs[2] = { num, b2 };
      uint32_t dB = bwpp_graph_add_op_node(grad, BWPP_GOP_DIV, dB_inputs, 2, &attr,
                                           &grad->values[num].shape,
                                           grad->values[num].dtype,
                                           grad->values[num].layout,
                                           0);
      dB = bwpp_graph_negate(grad, dB);
      dA = bwpp_graph_reduce_to_shape(grad, dA, &graph->values[n->inputs[0]].shape);
      dB = bwpp_graph_reduce_to_shape(grad, dB, &graph->values[n->inputs[1]].shape);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dA);
      grad_map[n->inputs[1]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[1]], dB);
      continue;
    }

    if (n->op == BWPP_GOP_TRANSPOSE && n->input_count >= 1) {
      uint32_t inputs[1] = { dY };
      BwppShape out_shape = grad->values[dY].shape;
      if (out_shape.rank == 2) {
        BwppStr tmp = out_shape.dims[0];
        out_shape.dims[0] = out_shape.dims[1];
        out_shape.dims[1] = tmp;
      }
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_TRANSPOSE, inputs, 1, NULL, &out_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    if (n->op == BWPP_GOP_PERMUTE && n->input_count >= 1) {
      BwppGraphAttr attr = n->attr;
      if (attr.perm_rank > 0 && attr.perm_rank <= BWPP_GRAPH_MAX_DIMS) {
        uint32_t inv[BWPP_GRAPH_MAX_DIMS] = {0};
        for (uint32_t k = 0; k < attr.perm_rank; ++k) {
          uint32_t src = attr.perm[k];
          if (src < attr.perm_rank) {
            inv[src] = k;
          }
        }
        for (uint32_t k = 0; k < attr.perm_rank; ++k) {
          attr.perm[k] = inv[k];
        }
      }
      uint32_t inputs[1] = { dY };
      BwppShape out_shape = grad->values[dY].shape;
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_PERMUTE, inputs, 1, &attr, &out_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    if (n->op == BWPP_GOP_RESHAPE && n->input_count >= 1) {
      BwppGraphAttr attr = {0};
      attr.shape = graph->values[n->inputs[0]].shape;
      uint32_t inputs[1] = { dY };
      BwppShape out_shape = graph->values[n->inputs[0]].shape;
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_RESHAPE, inputs, 1, &attr, &out_shape,
                                           grad->values[dY].dtype,
                                           grad->values[dY].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    if (n->op == BWPP_GOP_SILU && n->input_count >= 1) {
      uint32_t actX = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[0]);
      uint32_t inputs[2] = { actX, dY };
      BwppShape out_shape = grad->values[actX].shape;
      BwppGraphAttr attr = {0};
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_SILU_GRAD, inputs, 2, &attr, &out_shape,
                                           grad->values[actX].dtype,
                                           grad->values[actX].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    if (n->op == BWPP_GOP_SOFTMAX && n->input_count >= 1) {
      uint32_t actY = bwpp_graph_import_activation(grad, graph, act_map, n->output);
      uint32_t inputs[2] = { actY, dY };
      BwppShape out_shape = grad->values[actY].shape;
      BwppGraphAttr attr = n->attr;
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_SOFTMAX_GRAD, inputs, 2, &attr, &out_shape,
                                           grad->values[actY].dtype,
                                           grad->values[actY].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    if (n->op == BWPP_GOP_RMSNORM && n->input_count >= 2) {
      uint32_t actX = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[0]);
      uint32_t actGamma = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[1]);
      uint32_t inputs[3] = { actX, actGamma, dY };
      BwppShape out_shape = grad->values[actX].shape;
      BwppGraphAttr attr = n->attr;
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_RMSNORM_GRAD, inputs, 3, &attr, &out_shape,
                                           grad->values[actX].dtype,
                                           grad->values[actX].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);

      /* dGamma = reduce_sum(dY * (Y / gamma)) over non-gamma axes */
      uint32_t actY = bwpp_graph_import_activation(grad, graph, act_map, n->output);
      BwppGraphAttr math_attr = {0};
      uint32_t xhat = BWPP_GRAPH_NO_VALUE;
      if (n->input_count >= 3) {
        uint32_t actBeta = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[2]);
        uint32_t sub_inputs[2] = { actY, actBeta };
        uint32_t y_minus_beta = bwpp_graph_add_op_node(grad, BWPP_GOP_SUB, sub_inputs, 2, &math_attr,
                                                       &grad->values[actY].shape,
                                                       grad->values[actY].dtype,
                                                       grad->values[actY].layout,
                                                       0);
        uint32_t div_inputs[2] = { y_minus_beta, actGamma };
        xhat = bwpp_graph_add_op_node(grad, BWPP_GOP_DIV, div_inputs, 2, &math_attr,
                                      &grad->values[actY].shape,
                                      grad->values[actY].dtype,
                                      grad->values[actY].layout,
                                      0);
      } else {
        uint32_t div_inputs[2] = { actY, actGamma };
        xhat = bwpp_graph_add_op_node(grad, BWPP_GOP_DIV, div_inputs, 2, &math_attr,
                                      &grad->values[actY].shape,
                                      grad->values[actY].dtype,
                                      grad->values[actY].layout,
                                      0);
      }
      uint32_t mul_inputs[2] = { dY, xhat };
      uint32_t dgamma_raw = bwpp_graph_add_op_node(grad, BWPP_GOP_MUL, mul_inputs, 2, &math_attr,
                                                   &grad->values[dY].shape,
                                                   grad->values[dY].dtype,
                                                   grad->values[dY].layout,
                                                   0);
      BwppShape gamma_shape = graph->values[n->inputs[1]].shape;
      uint32_t dGamma = bwpp_graph_reduce_to_shape(grad, dgamma_raw, &gamma_shape);
      grad_map[n->inputs[1]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[1]], dGamma);

      if (n->input_count >= 3) {
        BwppShape beta_shape = graph->values[n->inputs[2]].shape;
        uint32_t dBeta = bwpp_graph_reduce_to_shape(grad, dY, &beta_shape);
        grad_map[n->inputs[2]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[2]], dBeta);
      }
      continue;
    }

    if (n->op == BWPP_GOP_REDUCE_SUM && n->input_count >= 1) {
      BwppShape in_shape = graph->values[n->inputs[0]].shape;
      uint32_t dX = bwpp_graph_broadcast_to_shape(grad, dY, &in_shape);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    if (n->op == BWPP_GOP_REDUCE_MAX && n->input_count >= 1) {
      uint32_t actX = bwpp_graph_import_activation(grad, graph, act_map, n->inputs[0]);
      BwppShape in_shape = graph->values[n->inputs[0]].shape;
      BwppGraphAttr attr = n->attr;
      if (!attr.has_axis) {
        attr.has_axis = 1;
        attr.axis = bwpp_shape_rank(&in_shape) - 1;
      }
      uint32_t mask_inputs[1] = { actX };
      uint32_t mask = bwpp_graph_add_op_node(grad, BWPP_GOP_REDUCE_MAX_MASK, mask_inputs, 1, &attr,
                                             &grad->values[actX].shape,
                                             grad->values[actX].dtype,
                                             grad->values[actX].layout,
                                             0);
      uint32_t dYb = bwpp_graph_broadcast_to_shape(grad, dY, &in_shape);
      uint32_t grad_inputs[2] = { mask, dYb };
      uint32_t dX = bwpp_graph_add_op_node(grad, BWPP_GOP_REDUCE_MAX_GRAD, grad_inputs, 2, &attr,
                                           &grad->values[actX].shape,
                                           grad->values[actX].dtype,
                                           grad->values[actX].layout,
                                           0);
      grad_map[n->inputs[0]] = bwpp_graph_accum_grad(grad, grad_map[n->inputs[0]], dX);
      continue;
    }

    fprintf(stderr, "autodiff: op %s not supported yet\n", bwpp_op_name(n->op));
  }

  for (uint32_t i = 0; i < graph->value_count; ++i) {
    if (graph->values[i].flags & BWPP_GRAPH_VALUE_INPUT) {
      uint32_t gval = grad_map[i];
      if (gval != BWPP_GRAPH_NO_VALUE) {
        grad->values[gval].flags |= BWPP_GRAPH_VALUE_OUTPUT;
        bwpp_graph_add_output(grad, gval);
      }
    }
  }

  free(act_map);
  free(grad_map);
  return grad;
}

void bwpp_graph_destroy(BwppGraph *graph) {
  if (!graph) {
    return;
  }
  free(graph->nodes);
  free(graph->values);
  free(graph->regions);
  free(graph->outputs);
  free(graph);
}

int bwpp_graph_detect_attention(const BwppGraph *graph) {
  if (!graph) {
    return 0;
  }
  for (uint32_t i = 0; i < graph->node_count; ++i) {
    const BwppGraphNode *mm = &graph->nodes[i];
    if (mm->op != BWPP_GOP_MATMUL || mm->input_count < 2) {
      continue;
    }
    int has_transpose = 0;
    for (uint32_t j = 0; j < mm->input_count; ++j) {
      uint32_t v = mm->inputs[j];
      if (v < graph->value_count) {
        uint32_t prod = graph->values[v].producer;
        if (prod != BWPP_GRAPH_NO_NODE && prod < graph->node_count &&
            graph->nodes[prod].op == BWPP_GOP_TRANSPOSE) {
          has_transpose = 1;
          break;
        }
      }
    }
    if (!has_transpose) {
      continue;
    }
    uint32_t scores = mm->output;
    uint32_t softmax_node = BWPP_GRAPH_NO_NODE;
    for (uint32_t j = 0; j < graph->node_count; ++j) {
      const BwppGraphNode *s = &graph->nodes[j];
      if (s->op == BWPP_GOP_SOFTMAX && s->input_count >= 1 && s->inputs[0] == scores) {
        softmax_node = j;
        break;
      }
    }
    if (softmax_node == BWPP_GRAPH_NO_NODE) {
      continue;
    }
    uint32_t probs = graph->nodes[softmax_node].output;
    for (uint32_t j = 0; j < graph->node_count; ++j) {
      const BwppGraphNode *mm2 = &graph->nodes[j];
      if (mm2->op == BWPP_GOP_MATMUL && mm2->input_count >= 2) {
        if (mm2->inputs[0] == probs || mm2->inputs[1] == probs) {
          return 1;
        }
      }
    }
  }
  return 0;
}

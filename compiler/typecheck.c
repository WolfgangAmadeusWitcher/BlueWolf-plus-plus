#include "typecheck.h"
#include "lexer.h"
#include <stdio.h>
#include <string.h>

#define BWPP_MAX_PARAMS 32
#define BWPP_MAX_DIMS 4

typedef struct {
  const char *ptr;
  size_t len;
} BwppStr;

typedef struct {
  BwppStr name;
  uint32_t rank;
  BwppStr dims[BWPP_MAX_DIMS];
} BwppParam;

typedef struct {
  uint32_t rank;
  BwppStr dims[BWPP_MAX_DIMS];
} BwppShape;

static int bwpp_str_eq(BwppStr s, const char *lit) {
  size_t len = strlen(lit);
  return s.len == len && strncmp(s.ptr, lit, len) == 0;
}

static int bwpp_str_eq_str(BwppStr a, BwppStr b) {
  return a.len == b.len && strncmp(a.ptr, b.ptr, a.len) == 0;
}

static int bwpp_tok_is(const BwppToken *tok, const char *lit) {
  size_t len = strlen(lit);
  return tok->kind == BWPP_TOK_IDENT && tok->length == len && strncmp(tok->lexeme, lit, len) == 0;
}

static BwppStr bwpp_tok_str(const BwppToken *tok) {
  BwppStr out = { tok->lexeme, tok->length };
  return out;
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

static int bwpp_parse_shape_list(BwppLexer *lx, BwppShape *shape) {
  BwppToken tok = bwpp_lexer_next(lx);
  if (!(tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '[')) {
    return 0;
  }
  shape->rank = 0;
  while (1) {
    BwppToken t = bwpp_lexer_next(lx);
    if (t.kind == BWPP_TOK_SYMBOL && t.length == 1 && t.lexeme[0] == ']') {
      break;
    }
    if (t.kind == BWPP_TOK_SYMBOL && t.length == 1 && t.lexeme[0] == ',') {
      continue;
    }
    if (shape->rank < BWPP_MAX_DIMS &&
        (t.kind == BWPP_TOK_IDENT || t.kind == BWPP_TOK_NUMBER)) {
      shape->dims[shape->rank++] = bwpp_tok_str(&t);
    }
  }
  return 1;
}

static int bwpp_parse_params(BwppLexer *lx, BwppParam *params, uint32_t *param_count) {
  BwppToken tok = bwpp_lexer_next(lx);
  if (!(tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '(')) {
    return 0;
  }
  while (1) {
    BwppToken name = bwpp_lexer_next(lx);
    if (name.kind == BWPP_TOK_SYMBOL && name.length == 1 && name.lexeme[0] == ')') {
      break;
    }
    if (name.kind != BWPP_TOK_IDENT) {
      return 0;
    }
    BwppToken colon = bwpp_lexer_next(lx);
    if (!(colon.kind == BWPP_TOK_SYMBOL && colon.length == 1 && colon.lexeme[0] == ':')) {
      return 0;
    }
    BwppToken tensor = bwpp_lexer_next(lx);
    if (!bwpp_tok_is(&tensor, "tensor")) {
      return 0;
    }
    BwppToken lt = bwpp_lexer_next(lx);
    if (!(lt.kind == BWPP_TOK_SYMBOL && lt.length == 1 && lt.lexeme[0] == '<')) {
      return 0;
    }
    BwppToken dtype = bwpp_lexer_next(lx);
    if (dtype.kind != BWPP_TOK_IDENT) {
      return 0;
    }
    BwppToken comma = bwpp_lexer_next(lx);
    if (!(comma.kind == BWPP_TOK_SYMBOL && comma.length == 1 && comma.lexeme[0] == ',')) {
      return 0;
    }
    BwppToken lbr = bwpp_lexer_next(lx);
    if (!(lbr.kind == BWPP_TOK_SYMBOL && lbr.length == 1 && lbr.lexeme[0] == '[')) {
      return 0;
    }

    if (*param_count >= BWPP_MAX_PARAMS) {
      return 0;
    }
    BwppParam *p = &params[*param_count];
    p->name = bwpp_tok_str(&name);
    p->rank = 0;

    while (1) {
      BwppToken dim = bwpp_lexer_next(lx);
      if (dim.kind == BWPP_TOK_SYMBOL && dim.length == 1 && dim.lexeme[0] == ']') {
        break;
      }
      if (dim.kind == BWPP_TOK_SYMBOL && dim.length == 1 && dim.lexeme[0] == ',') {
        continue;
      }
      if (p->rank < BWPP_MAX_DIMS && (dim.kind == BWPP_TOK_IDENT || dim.kind == BWPP_TOK_NUMBER)) {
        p->dims[p->rank++] = bwpp_tok_str(&dim);
      }
    }

    BwppToken comma2 = bwpp_lexer_next(lx);
    if (comma2.kind == BWPP_TOK_SYMBOL && comma2.length == 1 && comma2.lexeme[0] == ',') {
      BwppToken layout = bwpp_lexer_next(lx);
      (void)layout;
      BwppToken gt = bwpp_lexer_next(lx);
      if (!(gt.kind == BWPP_TOK_SYMBOL && gt.length == 1 && gt.lexeme[0] == '>')) {
        return 0;
      }
    } else if (comma2.kind == BWPP_TOK_SYMBOL && comma2.length == 1 && comma2.lexeme[0] == '>') {
      /* no layout */
    } else {
      return 0;
    }

    BwppToken sep = bwpp_lexer_next(lx);
    if (sep.kind == BWPP_TOK_SYMBOL && sep.length == 1 && sep.lexeme[0] == ')') {
      (*param_count)++;
      break;
    }
    if (!(sep.kind == BWPP_TOK_SYMBOL && sep.length == 1 && sep.lexeme[0] == ',')) {
      return 0;
    }
    (*param_count)++;
  }
  return 1;
}

static const BwppParam *bwpp_find_param(const BwppParam *params, uint32_t count, BwppStr name) {
  for (uint32_t i = 0; i < count; ++i) {
    if (params[i].name.len == name.len && strncmp(params[i].name.ptr, name.ptr, name.len) == 0) {
      return &params[i];
    }
  }
  return NULL;
}

BwppStatus bwpp_typecheck_module(const BwppAstModule *module) {
  if (!module || !module->source) {
    return BWPP_OK;
  }

  BwppLexer lx;
  bwpp_lexer_init(&lx, module->source, module->length);

  BwppParam params[BWPP_MAX_PARAMS];
  uint32_t param_count = 0;
  BwppStr matmul_out0 = {0};
  BwppStr matmul_out1 = {0};
  int saw_matmul = 0;
  int saw_bias_add = 0;
  BwppShape bias_shape = {0};
  int bias_shape_known = 0;

  BwppStr last_ident = (BwppStr){0};

  for (;;) {
    BwppToken tok = bwpp_lexer_next(&lx);
    if (tok.kind == BWPP_TOK_EOF) {
      break;
    }
    if (bwpp_tok_is(&tok, "fn")) {
      BwppToken name = bwpp_lexer_next(&lx);
      (void)name;
      if (!bwpp_parse_params(&lx, params, &param_count)) {
        fprintf(stderr, "typecheck: failed to parse params\n");
        return BWPP_ERR;
      }
      continue;
    }
    if (tok.kind == BWPP_TOK_IDENT && tok.length > 0) {
      last_ident = bwpp_tok_str(&tok);
      if (bwpp_tok_is(&tok, "add")) {
        BwppToken next = bwpp_lexer_next(&lx);
        if (next.kind == BWPP_TOK_SYMBOL && next.length == 1 && next.lexeme[0] == '(') {
          int depth = 1;
          BwppStr last = (BwppStr){0};
          while (depth > 0) {
            BwppToken arg = bwpp_lexer_next(&lx);
            if (arg.kind == BWPP_TOK_EOF) {
              break;
            }
            if (arg.kind == BWPP_TOK_SYMBOL && arg.length == 1) {
              if (arg.lexeme[0] == '@') {
                BwppToken rhs = bwpp_lexer_next(&lx);
                if (rhs.kind == BWPP_TOK_IDENT && last.ptr) {
                  const BwppParam *pa = bwpp_find_param(params, param_count, last);
                  const BwppParam *pb = bwpp_find_param(params, param_count, bwpp_tok_str(&rhs));
                  if (pa && pb && pa->rank == 2 && pb->rank == 2) {
                    if (!(pa->dims[1].len == pb->dims[0].len &&
                          strncmp(pa->dims[1].ptr, pb->dims[0].ptr, pb->dims[0].len) == 0)) {
                      fprintf(stderr, "typecheck: matmul K mismatch\n");
                      return BWPP_ERR;
                    }
                    matmul_out0 = pa->dims[0];
                    matmul_out1 = pb->dims[1];
                    saw_matmul = 1;
                  }
                }
              }
              if (arg.lexeme[0] == '(') {
                depth++;
              } else if (arg.lexeme[0] == ')') {
                depth--;
              }
              continue;
            }
            if (arg.kind == BWPP_TOK_IDENT) {
              last = bwpp_tok_str(&arg);
              BwppStr name = bwpp_tok_str(&arg);
              if (bwpp_str_eq(name, "reshape") || bwpp_str_eq(name, "permute")) {
                BwppToken open = bwpp_lexer_next(&lx);
                if (open.kind == BWPP_TOK_SYMBOL && open.length == 1 && open.lexeme[0] == '(') {
                  BwppToken inner = bwpp_lexer_next(&lx);
                  if (inner.kind == BWPP_TOK_IDENT && bwpp_tok_is(&inner, "bias")) {
                    const BwppParam *bias = bwpp_find_param(params, param_count, bwpp_tok_str(&inner));
                    if (!bias) {
                      fprintf(stderr, "typecheck: bias used but not declared\n");
                      return BWPP_ERR;
                    }
                    saw_bias_add = 1;
                    BwppToken comma = bwpp_lexer_next(&lx);
                    if (comma.kind == BWPP_TOK_SYMBOL && comma.length == 1 && comma.lexeme[0] == ',') {
                      BwppShape temp = {0};
                      if (!bwpp_parse_shape_list(&lx, &temp)) {
                        fprintf(stderr, "typecheck: reshape/permute missing shape list\n");
                        return BWPP_ERR;
                      }
                      if (bwpp_str_eq(name, "permute")) {
                        if (temp.rank != bias->rank) {
                          fprintf(stderr, "typecheck: permute rank mismatch for bias\n");
                          return BWPP_ERR;
                        }
                        int used[BWPP_MAX_DIMS] = {0};
                        bias_shape.rank = bias->rank;
                        for (uint32_t i = 0; i < temp.rank && i < BWPP_MAX_DIMS; ++i) {
                          uint32_t axis = 0;
                          if (!bwpp_str_to_u32(temp.dims[i], &axis)) {
                            fprintf(stderr, "typecheck: permute axes must be numeric\n");
                            return BWPP_ERR;
                          }
                          if (axis >= bias->rank) {
                            fprintf(stderr, "typecheck: permute axis out of range\n");
                            return BWPP_ERR;
                          }
                          if (used[axis]) {
                            fprintf(stderr, "typecheck: permute axis repeated\n");
                            return BWPP_ERR;
                          }
                          used[axis] = 1;
                          bias_shape.dims[i] = bias->dims[axis];
                        }
                      } else {
                        bias_shape = temp;
                      }
                      bias_shape_known = 1;
                    }
                  }
                }
              }
              if (bwpp_str_eq(name, "bias")) {
                saw_bias_add = 1;
                const BwppParam *bias = bwpp_find_param(params, param_count, name);
                if (!bias) {
                  fprintf(stderr, "typecheck: bias used but not declared\n");
                  return BWPP_ERR;
                }
                if (bias->rank != 1) {
                  fprintf(stderr, "typecheck: bias must be rank-1 tensor\n");
                  return BWPP_ERR;
                }
                bias_shape.rank = bias->rank;
                bias_shape.dims[0] = bias->dims[0];
                bias_shape_known = 1;
                if (saw_matmul && matmul_out1.ptr) {
                  if (!(bias->dims[0].len == matmul_out1.len &&
                        strncmp(bias->dims[0].ptr, matmul_out1.ptr, matmul_out1.len) == 0)) {
                    fprintf(stderr, "typecheck: bias shape does not match matmul N dimension\n");
                    return BWPP_ERR;
                  }
                }
              }
            }
          }
        }
      }
      if (bwpp_tok_is(&tok, "matmul")) {
        /* matmul(a, b) form */
        BwppToken next = bwpp_lexer_next(&lx);
        if (next.kind == BWPP_TOK_SYMBOL && next.length == 1 && next.lexeme[0] == '(') {
          BwppToken a = bwpp_lexer_next(&lx);
          BwppToken comma = bwpp_lexer_next(&lx);
          BwppToken b = bwpp_lexer_next(&lx);
          (void)comma;
          if (a.kind == BWPP_TOK_IDENT && b.kind == BWPP_TOK_IDENT) {
            const BwppParam *pa = bwpp_find_param(params, param_count, bwpp_tok_str(&a));
            const BwppParam *pb = bwpp_find_param(params, param_count, bwpp_tok_str(&b));
            if (pa && pb && pa->rank == 2 && pb->rank == 2) {
              if (!(pa->dims[1].len == pb->dims[0].len &&
                    strncmp(pa->dims[1].ptr, pb->dims[0].ptr, pb->dims[0].len) == 0)) {
                fprintf(stderr, "typecheck: matmul K mismatch\n");
                return BWPP_ERR;
              }
              matmul_out0 = pa->dims[0];
              matmul_out1 = pb->dims[1];
              saw_matmul = 1;
            }
          }
        }
      }
    }
    if (tok.kind == BWPP_TOK_SYMBOL && tok.length == 1 && tok.lexeme[0] == '@') {
      BwppToken rhs = bwpp_lexer_next(&lx);
      if (rhs.kind == BWPP_TOK_IDENT && last_ident.ptr) {
        const BwppParam *pa = bwpp_find_param(params, param_count, last_ident);
        const BwppParam *pb = bwpp_find_param(params, param_count, bwpp_tok_str(&rhs));
        if (pa && pb && pa->rank == 2 && pb->rank == 2) {
          if (!(pa->dims[1].len == pb->dims[0].len &&
                strncmp(pa->dims[1].ptr, pb->dims[0].ptr, pb->dims[0].len) == 0)) {
            fprintf(stderr, "typecheck: matmul K mismatch\n");
            return BWPP_ERR;
          }
          matmul_out0 = pa->dims[0];
          matmul_out1 = pb->dims[1];
          saw_matmul = 1;
        }
      }
    }
  }

  if (saw_bias_add && !saw_matmul) {
    fprintf(stderr, "typecheck: add(bias) without matmul context\n");
    return BWPP_ERR;
  }
  if (saw_bias_add && saw_matmul && bias_shape_known && matmul_out1.ptr) {
    int ok = 0;
    if (bias_shape.rank == 1) {
      ok = bwpp_str_eq_str(bias_shape.dims[0], matmul_out1);
    } else if (bias_shape.rank == 2) {
      int d0_is_one = bwpp_str_eq(bias_shape.dims[0], "1");
      int d1_is_one = bwpp_str_eq(bias_shape.dims[1], "1");
      if ((d0_is_one && bwpp_str_eq_str(bias_shape.dims[1], matmul_out1)) ||
          (d1_is_one && bwpp_str_eq_str(bias_shape.dims[0], matmul_out1))) {
        ok = 1;
      }
    }
    if (!ok) {
      fprintf(stderr, "typecheck: bias shape does not match matmul N dimension\n");
      return BWPP_ERR;
    }
  }
  return BWPP_OK;
}

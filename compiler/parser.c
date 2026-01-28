#include "parser.h"
#include <ctype.h>
#include <string.h>

static BwppToken bwpp_parser_next(BwppParser *parser);
static void bwpp_parser_unread(BwppParser *parser, BwppToken tok);

void bwpp_parser_init(BwppParser *parser, const char *input, size_t length) {
  bwpp_lexer_init(&parser->lexer, input, length);
  parser->source = input;
  parser->length = length;
  parser->has_lookahead = 0;
}

static BwppToken bwpp_parser_next(BwppParser *parser) {
  if (parser->has_lookahead) {
    parser->has_lookahead = 0;
    return parser->lookahead;
  }
  return bwpp_lexer_next(&parser->lexer);
}

static void bwpp_parser_unread(BwppParser *parser, BwppToken tok) {
  parser->lookahead = tok;
  parser->has_lookahead = 1;
}

static int bwpp_token_is(const BwppToken *tok, const char *lit) {
  size_t len = strlen(lit);
  return tok->kind == BWPP_TOK_IDENT && tok->length == len && strncmp(tok->lexeme, lit, len) == 0;
}

BwppAstModule *bwpp_parse_module(BwppParser *parser) {
  BwppAstModule *module = bwpp_ast_module_create(parser->source, parser->length);
  if (!module) {
    return NULL;
  }

  int brace_depth = 0;
  int reversible_brace_depth = -1;
  uint32_t current_region = BWPP_AST_NO_REGION;
  int pending_reversible = 0;
  int pending_reversible_fn = 0;
  int expect_fn_name = 0;

  for (;;) {
    BwppToken tok = bwpp_parser_next(parser);
    if (tok.kind == BWPP_TOK_EOF) {
      break;
    }
    if (tok.kind == BWPP_TOK_SYMBOL && tok.length == 1) {
      char ch = tok.lexeme[0];
      if (ch == '{') {
        brace_depth++;
        if (pending_reversible_fn && current_region == BWPP_AST_NO_REGION) {
          current_region = bwpp_ast_add_region(module, BWPP_AST_REGION_REVERSIBLE, BWPP_AST_POLICY_AUTO);
          reversible_brace_depth = brace_depth;
          pending_reversible_fn = 0;
          pending_reversible = 0;
        }
        continue;
      }
      if (ch == '}') {
        if (current_region != BWPP_AST_NO_REGION && reversible_brace_depth == brace_depth) {
          current_region = BWPP_AST_NO_REGION;
          reversible_brace_depth = -1;
        }
        brace_depth--;
        continue;
      }
      if (ch == '@') {
        BwppToken next = bwpp_parser_next(parser);
        if (bwpp_token_is(&next, "reversible")) {
          pending_reversible = 1;
          continue;
        }
        if (bwpp_token_is(&next, "meta") || bwpp_token_is(&next, "impure")) {
          continue;
        }
        if (brace_depth > 0) {
          bwpp_ast_add_op(module, BWPP_AST_OP_MATMUL, current_region, 0);
        }
        bwpp_parser_unread(parser, next);
        continue;
      }
      continue;
    }

    if (tok.kind == BWPP_TOK_IDENT) {
      if (bwpp_token_is(&tok, "fn")) {
        expect_fn_name = 1;
        if (pending_reversible) {
          pending_reversible_fn = 1;
        }
        continue;
      }
      if (expect_fn_name) {
        expect_fn_name = 0;
        continue;
      }
      if (brace_depth <= 0) {
        continue;
      }
      if (bwpp_token_is(&tok, "matmul")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_MATMUL, current_region, 0);
      } else if (bwpp_token_is(&tok, "batch_matmul")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_BATCH_MATMUL, current_region, 0);
      } else if (bwpp_token_is(&tok, "transpose")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_TRANSPOSE, current_region, 0);
      } else if (bwpp_token_is(&tok, "permute")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_PERMUTE, current_region, 0);
      } else if (bwpp_token_is(&tok, "reshape")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_RESHAPE, current_region, 0);
      } else if (bwpp_token_is(&tok, "add")) {
        uint32_t flags = 0;
        BwppToken next = bwpp_parser_next(parser);
        if (next.kind == BWPP_TOK_SYMBOL && next.length == 1 && next.lexeme[0] == '(') {
          int depth = 1;
          while (depth > 0) {
            BwppToken arg = bwpp_parser_next(parser);
            if (arg.kind == BWPP_TOK_EOF) {
              break;
            }
            if (arg.kind == BWPP_TOK_SYMBOL && arg.length == 1) {
              if (arg.lexeme[0] == '@') {
                bwpp_ast_add_op(module, BWPP_AST_OP_MATMUL, current_region, 0);
              }
              if (arg.lexeme[0] == '(') {
                depth++;
              } else if (arg.lexeme[0] == ')') {
                depth--;
              }
              continue;
            }
            if (arg.kind == BWPP_TOK_IDENT && bwpp_token_is(&arg, "matmul")) {
              bwpp_ast_add_op(module, BWPP_AST_OP_MATMUL, current_region, 0);
            }
            if (arg.kind == BWPP_TOK_IDENT && bwpp_token_is(&arg, "bias")) {
              flags |= BWPP_AST_OPF_HAS_BIAS;
            }
          }
        } else {
          bwpp_parser_unread(parser, next);
        }
        bwpp_ast_add_op(module, BWPP_AST_OP_ADD, current_region, flags);
      } else if (bwpp_token_is(&tok, "sub")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_SUB, current_region, 0);
      } else if (bwpp_token_is(&tok, "mul")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_MUL, current_region, 0);
      } else if (bwpp_token_is(&tok, "div")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_DIV, current_region, 0);
      } else if (bwpp_token_is(&tok, "reduce_sum")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_REDUCE_SUM, current_region, 0);
      } else if (bwpp_token_is(&tok, "reduce_max")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_REDUCE_MAX, current_region, 0);
      } else if (bwpp_token_is(&tok, "softmax")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_SOFTMAX, current_region, 0);
      } else if (bwpp_token_is(&tok, "rmsnorm")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_RMSNORM, current_region, 0);
      } else if (bwpp_token_is(&tok, "silu")) {
        bwpp_ast_add_op(module, BWPP_AST_OP_SILU, current_region, 0);
      }
    }
  }

  return module;
}

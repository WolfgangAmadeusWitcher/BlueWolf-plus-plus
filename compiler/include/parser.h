#ifndef BWPP_PARSER_H
#define BWPP_PARSER_H

#include "ast.h"
#include "lexer.h"

typedef struct {
  BwppLexer lexer;
  const char *source;
  size_t length;
  int has_lookahead;
  BwppToken lookahead;
} BwppParser;

void bwpp_parser_init(BwppParser *parser, const char *input, size_t length);
BwppAstModule *bwpp_parse_module(BwppParser *parser);

#endif

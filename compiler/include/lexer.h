#ifndef BWPP_LEXER_H
#define BWPP_LEXER_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
  BWPP_TOK_EOF = 0,
  BWPP_TOK_ERROR,
  BWPP_TOK_IDENT,
  BWPP_TOK_NUMBER,
  BWPP_TOK_SYMBOL
} BwppTokenKind;

typedef struct {
  BwppTokenKind kind;
  const char *lexeme;
  size_t length;
} BwppToken;

typedef struct {
  const char *input;
  size_t length;
  size_t pos;
} BwppLexer;

void bwpp_lexer_init(BwppLexer *lx, const char *input, size_t length);
BwppToken bwpp_lexer_next(BwppLexer *lx);

#endif

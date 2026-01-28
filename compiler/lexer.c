#include "lexer.h"
#include <ctype.h>

static BwppToken bwpp_make_token(BwppTokenKind kind, const char *start, size_t length) {
  BwppToken tok;
  tok.kind = kind;
  tok.lexeme = start;
  tok.length = length;
  return tok;
}

void bwpp_lexer_init(BwppLexer *lx, const char *input, size_t length) {
  lx->input = input;
  lx->length = length;
  lx->pos = 0;
}

static void bwpp_skip_ws(BwppLexer *lx) {
  while (lx->pos < lx->length) {
    char c = lx->input[lx->pos];
    if (isspace((unsigned char)c)) {
      lx->pos++;
      continue;
    }
    if (c == '/' && lx->pos + 1 < lx->length) {
      char n = lx->input[lx->pos + 1];
      if (n == '/') {
        lx->pos += 2;
        while (lx->pos < lx->length && lx->input[lx->pos] != '\n') {
          lx->pos++;
        }
        continue;
      }
      if (n == '*') {
        lx->pos += 2;
        while (lx->pos + 1 < lx->length &&
               !(lx->input[lx->pos] == '*' && lx->input[lx->pos + 1] == '/')) {
          lx->pos++;
        }
        if (lx->pos + 1 < lx->length) {
          lx->pos += 2;
        }
        continue;
      }
    }
    break;
  }
}

BwppToken bwpp_lexer_next(BwppLexer *lx) {
  if (!lx) {
    return bwpp_make_token(BWPP_TOK_EOF, NULL, 0);
  }
  bwpp_skip_ws(lx);
  if (lx->pos >= lx->length) {
    return bwpp_make_token(BWPP_TOK_EOF, lx->input + lx->length, 0);
  }

  const char *start = lx->input + lx->pos;
  unsigned char c = (unsigned char)lx->input[lx->pos];
  if (isalpha(c) || c == '_') {
    lx->pos++;
    while (lx->pos < lx->length) {
      unsigned char ch = (unsigned char)lx->input[lx->pos];
      if (!(isalnum(ch) || ch == '_')) {
        break;
      }
      lx->pos++;
    }
    size_t len = (size_t)(lx->input + lx->pos - start);
    return bwpp_make_token(BWPP_TOK_IDENT, start, len);
  }
  if (isdigit(c)) {
    lx->pos++;
    while (lx->pos < lx->length && isdigit((unsigned char)lx->input[lx->pos])) {
      lx->pos++;
    }
    size_t len = (size_t)(lx->input + lx->pos - start);
    return bwpp_make_token(BWPP_TOK_NUMBER, start, len);
  }

  lx->pos++;
  return bwpp_make_token(BWPP_TOK_SYMBOL, start, 1);
}

#include "codegen_metal.h"
#include "ir.h"
#include "parser.h"
#include "typecheck.h"
#include <stdio.h>
#include <stdlib.h>

static char *bwpp_read_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long size = ftell(f);
  if (size < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  char *buf = (char *)malloc((size_t)size + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t read = fread(buf, 1, (size_t)size, f);
  fclose(f);
  buf[read] = '\0';
  if (out_len) {
    *out_len = read;
  }
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <input.bwpp> <output.metal>\n", argv[0]);
    return 1;
  }

  size_t len = 0;
  char *src = bwpp_read_file(argv[1], &len);
  if (!src) {
    fprintf(stderr, "failed to read input: %s\n", argv[1]);
    return 1;
  }

  BwppParser parser;
  bwpp_parser_init(&parser, src, len);
  BwppAstModule *module = bwpp_parse_module(&parser);
  if (!module) {
    fprintf(stderr, "parse failed\n");
    free(src);
    return 1;
  }

  if (bwpp_typecheck_module(module) != BWPP_OK) {
    fprintf(stderr, "typecheck failed\n");
    bwpp_ast_module_destroy(module);
    free(src);
    return 1;
  }

  BwppIrModule *ir = bwpp_ir_from_ast(module);
  if (!ir) {
    fprintf(stderr, "ir failed\n");
    bwpp_ast_module_destroy(module);
    free(src);
    return 1;
  }

  if (bwpp_codegen_metal(ir, argv[2]) != BWPP_OK) {
    fprintf(stderr, "codegen failed\n");
    bwpp_ir_destroy(ir);
    bwpp_ast_module_destroy(module);
    free(src);
    return 1;
  }

  bwpp_ir_destroy(ir);
  bwpp_ast_module_destroy(module);
  free(src);
  return 0;
}

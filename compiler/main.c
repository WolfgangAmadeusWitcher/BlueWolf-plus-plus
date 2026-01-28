#include "codegen_metal.h"
#include "graph_ir.h"
#include "ir.h"
#include "mem_plan.h"
#include "parser.h"
#include "typecheck.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  const char *input_path = NULL;
  const char *output_path = NULL;
  const char *dot_path = NULL;
  const char *grad_dot_path = NULL;
  const char *mem_plan_path = NULL;
  int attn_report = 0;
  const char *entry = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--dot") == 0 && i + 1 < argc) {
      dot_path = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--grad-dot") == 0 && i + 1 < argc) {
      grad_dot_path = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--mem-plan") == 0 && i + 1 < argc) {
      mem_plan_path = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
      entry = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--attn-report") == 0) {
      attn_report = 1;
      continue;
    }
    if (!input_path) {
      input_path = argv[i];
    } else if (!output_path) {
      output_path = argv[i];
    } else {
      fprintf(stderr, "unexpected arg: %s\n", argv[i]);
      return 1;
    }
  }

  if (!input_path || !output_path) {
    fprintf(stderr,
            "usage: %s <input.bwpp> <output.metal> [--dot <graph.dot>] [--grad-dot <grad.dot>]\n"
            "       [--mem-plan <plan.txt>] [--attn-report] [--entry <fn>]\n",
            argv[0]);
    return 1;
  }

  size_t len = 0;
  char *src = bwpp_read_file(input_path, &len);
  if (!src) {
    fprintf(stderr, "failed to read input: %s\n", input_path);
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

  BwppGraph *graph = bwpp_graph_build(module, entry);
  if (!graph) {
    fprintf(stderr, "graph build failed");
    if (entry && entry[0] != '\0') {
      fprintf(stderr, " (entry %s not found)\n", entry);
    } else {
      fprintf(stderr, "\n");
    }
    bwpp_ast_module_destroy(module);
    free(src);
    return 1;
  }

  BwppIrModule *ir = bwpp_ir_from_graph(graph);
  if (!ir) {
    fprintf(stderr, "ir failed\n");
    bwpp_graph_destroy(graph);
    bwpp_ast_module_destroy(module);
    free(src);
    return 1;
  }

  if (dot_path && graph) {
    FILE *dot = fopen(dot_path, "w");
    if (!dot) {
      fprintf(stderr, "failed to open dot output: %s\n", dot_path);
    } else {
      bwpp_graph_dump_dot(graph, dot);
      fclose(dot);
    }
  }

  if (grad_dot_path && graph) {
    BwppGraph *grad = bwpp_graph_autodiff(graph);
    if (!grad) {
      fprintf(stderr, "failed to build autodiff graph\n");
    } else {
      FILE *dot = fopen(grad_dot_path, "w");
      if (!dot) {
        fprintf(stderr, "failed to open grad dot output: %s\n", grad_dot_path);
      } else {
        bwpp_graph_dump_dot(grad, dot);
        fclose(dot);
      }
      bwpp_graph_destroy(grad);
    }
  }

  int has_attention = 0;
  if (graph) {
    has_attention = bwpp_graph_detect_attention(graph);
    if (has_attention) {
      ir->flags |= BWPP_IRF_HAS_ATTENTION;
    }
    if (attn_report) {
      fprintf(stderr, "attention_candidate=%d\n", has_attention);
    }
  }

  if (mem_plan_path && graph) {
    BwppMemPlan *plan = bwpp_mem_plan_build(graph);
    if (!plan) {
      fprintf(stderr, "failed to build mem plan\n");
    } else {
      FILE *out = fopen(mem_plan_path, "w");
      if (!out) {
        fprintf(stderr, "failed to open mem plan output: %s\n", mem_plan_path);
      } else {
        bwpp_mem_plan_dump(plan, out);
        fclose(out);
      }
      bwpp_mem_plan_destroy(plan);
    }
  }

  if (bwpp_codegen_metal(ir, output_path) != BWPP_OK) {
    fprintf(stderr, "codegen failed\n");
    bwpp_graph_destroy(graph);
    bwpp_ir_destroy(ir);
    bwpp_ast_module_destroy(module);
    free(src);
    return 1;
  }

  bwpp_graph_destroy(graph);
  bwpp_ir_destroy(ir);
  bwpp_ast_module_destroy(module);
  free(src);
  return 0;
}

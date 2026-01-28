#include "codegen_metal.h"
#include "tile_ir.h"
#include <stdio.h>

static BwppTileKernel *bwpp_lower_tile_matmul(const BwppIrModule *ir) {
  if (!ir) {
    return NULL;
  }
  int has_matmul = 0;
  int has_add = 0;
  int has_silu = 0;
  for (uint32_t i = 0; i < ir->node_count; ++i) {
    if (ir->nodes[i].op == BWPP_OP_MATMUL) {
      has_matmul = 1;
    } else if (ir->nodes[i].op == BWPP_OP_ADD) {
      if (ir->nodes[i].flags & BWPP_IR_OPF_HAS_BIAS) {
        has_add = 1;
      }
    } else if (ir->nodes[i].op == BWPP_OP_SILU) {
      has_silu = 1;
    }
  }
  if (!has_matmul) {
    return NULL;
  }
  BwppTileKernel *kernel = bwpp_tile_kernel_create();
  if (!kernel) {
    return NULL;
  }
  kernel->block.m = 128;
  kernel->block.n = 128;
  kernel->block.k = 32;
  BwppTileOp load_a;
  load_a.kind = BWPP_TILE_OP_LOAD;
  load_a.tile.m = 16;
  load_a.tile.n = 16;
  load_a.tile.k = 16;
  load_a.src_mem = BWPP_TILE_MEM_GLOBAL;
  load_a.dst_mem = BWPP_TILE_MEM_THREADGROUP;
  load_a.role = BWPP_TILE_ROLE_A;
  load_a.epilogue = BWPP_TILE_EPILOGUE_NONE;
  load_a.a_mem = BWPP_TILE_MEM_GLOBAL;
  load_a.b_mem = BWPP_TILE_MEM_GLOBAL;
  load_a.c_mem = BWPP_TILE_MEM_GLOBAL;
  if (bwpp_tile_kernel_add_op(kernel, &load_a) != BWPP_OK) {
    bwpp_tile_kernel_destroy(kernel);
    return NULL;
  }

  BwppTileOp load_b = load_a;
  load_b.role = BWPP_TILE_ROLE_B;
  if (bwpp_tile_kernel_add_op(kernel, &load_b) != BWPP_OK) {
    bwpp_tile_kernel_destroy(kernel);
    return NULL;
  }

  BwppTileOp op;
  op.kind = BWPP_TILE_OP_MATMUL;
  op.tile.m = 16;
  op.tile.n = 16;
  op.tile.k = 16;
  op.a_mem = BWPP_TILE_MEM_THREADGROUP;
  op.b_mem = BWPP_TILE_MEM_THREADGROUP;
  op.c_mem = BWPP_TILE_MEM_REGISTER;
  op.src_mem = BWPP_TILE_MEM_THREADGROUP;
  op.dst_mem = BWPP_TILE_MEM_REGISTER;
  op.role = BWPP_TILE_ROLE_C;
  op.epilogue = BWPP_TILE_EPILOGUE_NONE;
  if (bwpp_tile_kernel_add_op(kernel, &op) != BWPP_OK) {
    bwpp_tile_kernel_destroy(kernel);
    return NULL;
  }
  if (has_add || has_silu) {
    BwppTileOp epi;
    epi.kind = BWPP_TILE_OP_ELEMENTWISE;
    epi.tile = op.tile;
    epi.a_mem = BWPP_TILE_MEM_REGISTER;
    epi.b_mem = BWPP_TILE_MEM_REGISTER;
    epi.c_mem = BWPP_TILE_MEM_REGISTER;
    epi.src_mem = BWPP_TILE_MEM_REGISTER;
    epi.dst_mem = BWPP_TILE_MEM_REGISTER;
    epi.role = BWPP_TILE_ROLE_C;
    if (has_add && has_silu) {
      epi.epilogue = BWPP_TILE_EPILOGUE_ADD_SILU;
    } else if (has_add) {
      epi.epilogue = BWPP_TILE_EPILOGUE_ADD;
    } else {
      epi.epilogue = BWPP_TILE_EPILOGUE_SILU;
    }
    if (bwpp_tile_kernel_add_op(kernel, &epi) != BWPP_OK) {
      bwpp_tile_kernel_destroy(kernel);
      return NULL;
    }
  }

  BwppTileOp store_c;
  store_c.kind = BWPP_TILE_OP_STORE;
  store_c.tile = op.tile;
  store_c.src_mem = BWPP_TILE_MEM_REGISTER;
  store_c.dst_mem = BWPP_TILE_MEM_GLOBAL;
  store_c.role = BWPP_TILE_ROLE_C;
  store_c.epilogue = BWPP_TILE_EPILOGUE_NONE;
  store_c.a_mem = BWPP_TILE_MEM_REGISTER;
  store_c.b_mem = BWPP_TILE_MEM_REGISTER;
  store_c.c_mem = BWPP_TILE_MEM_GLOBAL;
  if (bwpp_tile_kernel_add_op(kernel, &store_c) != BWPP_OK) {
    bwpp_tile_kernel_destroy(kernel);
    return NULL;
  }
  return kernel;
}

static BwppTileKernel *bwpp_lower_tile_attention_stub(void) {
  BwppTileKernel *kernel = bwpp_tile_kernel_create();
  if (!kernel) {
    return NULL;
  }
  kernel->block.m = 128;
  kernel->block.n = 128;
  kernel->block.k = 32;
  BwppTileOp op;
  op.kind = BWPP_TILE_OP_ATTENTION;
  op.tile.m = 16;
  op.tile.n = 16;
  op.tile.k = 16;
  op.a_mem = BWPP_TILE_MEM_THREADGROUP;
  op.b_mem = BWPP_TILE_MEM_THREADGROUP;
  op.c_mem = BWPP_TILE_MEM_REGISTER;
  op.src_mem = BWPP_TILE_MEM_THREADGROUP;
  op.dst_mem = BWPP_TILE_MEM_REGISTER;
  op.role = BWPP_TILE_ROLE_C;
  op.epilogue = BWPP_TILE_EPILOGUE_NONE;
  if (bwpp_tile_kernel_add_op(kernel, &op) != BWPP_OK) {
    bwpp_tile_kernel_destroy(kernel);
    return NULL;
  }
  return kernel;
}

BwppStatus bwpp_codegen_metal(const BwppIrModule *ir, const char *out_path) {
  FILE *f = fopen(out_path, "w");
  if (!f) {
    return BWPP_ERR;
  }
  uint32_t op_count = ir ? ir->node_count : 0;
  uint32_t region_count = ir ? ir->region_count : 0;
  int has_softmax = 0;
  int has_rmsnorm = 0;
  if (ir) {
    for (uint32_t i = 0; i < ir->node_count; ++i) {
      if (ir->nodes[i].op == BWPP_OP_SOFTMAX) {
        has_softmax = 1;
      } else if (ir->nodes[i].op == BWPP_OP_RMSNORM) {
        has_rmsnorm = 1;
      }
    }
  }
  int has_attention = ir && (ir->flags & BWPP_IRF_HAS_ATTENTION);
  BwppTileKernel *tile = has_attention ? bwpp_lower_tile_attention_stub() : bwpp_lower_tile_matmul(ir);
  const BwppTileOp *matmul = NULL;
  const BwppTileOp *epi = NULL;
  uint32_t tile_m = 16;
  uint32_t tile_n = 16;
  uint32_t tile_k = 16;
  if (tile) {
    for (uint32_t i = 0; i < tile->op_count; ++i) {
      if (tile->ops[i].kind == BWPP_TILE_OP_MATMUL) {
        matmul = &tile->ops[i];
        tile_m = matmul->tile.m;
        tile_n = matmul->tile.n;
        tile_k = matmul->tile.k;
        break;
      }
    }
    for (uint32_t i = 0; i < tile->op_count; ++i) {
      if (tile->ops[i].kind == BWPP_TILE_OP_ELEMENTWISE) {
        epi = &tile->ops[i];
        break;
      }
    }
  }
  fputs("// BW++ Metal output stub\n", f);
  fprintf(f, "// bwpp.meta: ops=%u reversible_regions=%u\n", op_count, region_count);
  const char *policy = "auto";
  int has_store = 0;
  int has_recompute = 0;
  int has_auto = 0;
  for (uint32_t i = 0; i < region_count; ++i) {
    BwppRegionPolicy p = ir->regions[i].policy;
    if (p == BWPP_POLICY_STORE) {
      has_store = 1;
    } else if (p == BWPP_POLICY_RECOMPUTE) {
      has_recompute = 1;
    } else {
      has_auto = 1;
    }
  }
  if (has_store && !has_recompute && !has_auto) {
    policy = "store";
  } else if (has_recompute && !has_store && !has_auto) {
    policy = "recompute";
  } else if (has_store || has_recompute) {
    policy = "mixed";
  }
  fprintf(f, "// bwpp.meta: reversible_policy=%s\n", policy);
  for (uint32_t i = 0; i < region_count; ++i) {
    const BwppIrRegion *r = &ir->regions[i];
    const char *kind = r->kind == BWPP_REGION_REVERSIBLE ? "reversible" : "normal";
    const char *pol = "auto";
    if (r->policy == BWPP_POLICY_STORE) {
      pol = "store";
    } else if (r->policy == BWPP_POLICY_RECOMPUTE) {
      pol = "recompute";
    }
    fprintf(f, "// bwpp.meta: region=%u kind=%s policy=%s\n", r->id, kind, pol);
  }
  if (tile) {
    if (has_attention) {
      fputs("// bwpp.meta: kernel=attention_f16\n", f);
      fputs("// bwpp.meta: attention_plan=tile_ir_stub\n", f);
      fputs("// bwpp.meta: fused_attention_candidate=1\n", f);
    } else {
      fputs("// bwpp.meta: kernel=matmul_f16\n", f);
    }
    fputs("// bwpp.meta: layout=row_major\n", f);
    fprintf(f, "// bwpp.meta: block=%u,%u,%u\n", tile->block.m, tile->block.n, tile->block.k);
    if (matmul) {
      fprintf(f, "// bwpp.meta: tile=%u,%u,%u\n", tile_m, tile_n, tile_k);
    }
    if (epi) {
      const char *ep = "none";
      if (epi->epilogue == BWPP_TILE_EPILOGUE_ADD) {
        ep = "add";
      } else if (epi->epilogue == BWPP_TILE_EPILOGUE_SILU) {
        ep = "silu";
      } else if (epi->epilogue == BWPP_TILE_EPILOGUE_ADD_SILU) {
        ep = "add_silu";
      }
      fprintf(f, "// bwpp.meta: epilogue=%s\n", ep);
    }
    fputs("// bwpp.meta: params=M,N,K,lda,ldb,ldc\n\n", f);
  } else {
    fputs("// bwpp.meta: kernel=none\n\n", f);
  }
  if (has_softmax) {
    fputs("// bwpp.meta: aux_kernel=softmax_f16\n", f);
  }
  if (has_rmsnorm) {
    fputs("// bwpp.meta: aux_kernel=rmsnorm_f16\n", f);
  }
  if (tile) {
    fputs("#include <metal_stdlib>\n", f);
    fputs("using namespace metal;\n\n", f);
    if (!has_attention) {
      fprintf(f, "#define TILE_M %u\n", tile_m);
      fprintf(f, "#define TILE_N %u\n", tile_n);
      fprintf(f, "#define TILE_K %u\n\n", tile_k);
      int ep_add = 0;
      int ep_silu = 0;
      if (epi) {
        if (epi->epilogue == BWPP_TILE_EPILOGUE_ADD) {
          ep_add = 1;
        } else if (epi->epilogue == BWPP_TILE_EPILOGUE_SILU) {
          ep_silu = 1;
        } else if (epi->epilogue == BWPP_TILE_EPILOGUE_ADD_SILU) {
          ep_add = 1;
          ep_silu = 1;
        }
      }
      fprintf(f, "#define BWPP_EPILOGUE_ADD %d\n", ep_add);
      fprintf(f, "#define BWPP_EPILOGUE_SILU %d\n\n", ep_silu);
      fputs("struct BwppMatmulParams {\n", f);
      fputs("  uint M;\n", f);
      fputs("  uint N;\n", f);
      fputs("  uint K;\n", f);
      fputs("  uint lda;\n", f);
      fputs("  uint ldb;\n", f);
      fputs("  uint ldc;\n", f);
      fputs("};\n\n", f);
      fputs("inline float bwpp_silu(float x) {\n", f);
      fputs("  return x / (1.0f + exp(-x));\n", f);
      fputs("}\n\n", f);
      fputs("kernel void bwpp_matmul_f16(\n", f);
      fputs("    device const half *A [[buffer(0)]],\n", f);
      fputs("    device const half *B [[buffer(1)]],\n", f);
      fputs("    device half *C [[buffer(2)]],\n", f);
      fputs("    constant BwppMatmulParams &p [[buffer(3)]],\n", f);
      fputs("    device const half *Bias [[buffer(4)]],\n", f);
      fputs("    uint2 tid [[thread_position_in_threadgroup]],\n", f);
      fputs("    uint2 tgid [[threadgroup_position_in_grid]]) {\n", f);
      fputs("  threadgroup half As[TILE_M][TILE_K];\n", f);
      fputs("  threadgroup half Bs[TILE_K][TILE_N];\n", f);
      fputs("  uint row = tgid.y * TILE_M + tid.y;\n", f);
      fputs("  uint col = tgid.x * TILE_N + tid.x;\n", f);
      fputs("  float acc = 0.0f;\n", f);
      fputs("  for (uint k0 = 0; k0 < p.K; k0 += TILE_K) {\n", f);
      fputs("    uint a_col = k0 + tid.x;\n", f);
      fputs("    if (row < p.M && a_col < p.K) {\n", f);
      fputs("      As[tid.y][tid.x] = A[row * p.lda + a_col];\n", f);
      fputs("    } else {\n", f);
      fputs("      As[tid.y][tid.x] = half(0.0f);\n", f);
      fputs("    }\n", f);
      fputs("    uint b_row = k0 + tid.y;\n", f);
      fputs("    if (b_row < p.K && col < p.N) {\n", f);
      fputs("      Bs[tid.y][tid.x] = B[b_row * p.ldb + col];\n", f);
      fputs("    } else {\n", f);
      fputs("      Bs[tid.y][tid.x] = half(0.0f);\n", f);
      fputs("    }\n", f);
      fputs("    threadgroup_barrier(mem_flags::mem_threadgroup);\n", f);
      fputs("    for (uint k = 0; k < TILE_K; ++k) {\n", f);
      fputs("      acc += float(As[tid.y][k]) * float(Bs[k][tid.x]);\n", f);
      fputs("    }\n", f);
      fputs("    threadgroup_barrier(mem_flags::mem_threadgroup);\n", f);
      fputs("  }\n", f);
      fputs("  if (row < p.M && col < p.N) {\n", f);
      fputs("    float out = acc;\n", f);
      fputs("#if BWPP_EPILOGUE_ADD\n", f);
      fputs("    out += float(Bias[col]);\n", f);
      fputs("#endif\n", f);
      fputs("#if BWPP_EPILOGUE_SILU\n", f);
      fputs("    out = bwpp_silu(out);\n", f);
      fputs("#endif\n", f);
      fputs("    C[row * p.ldc + col] = half(out);\n", f);
      fputs("  }\n", f);
      fputs("}\n", f);
    } else {
      fputs("struct BwppAttentionParams {\n", f);
      fputs("  uint M;\n", f);
      fputs("  uint N;\n", f);
      fputs("  uint K;\n", f);
      fputs("};\n\n", f);
      fputs("kernel void bwpp_attention_f16(\n", f);
      fputs("    device const half *Q [[buffer(0)]],\n", f);
      fputs("    device const half *K [[buffer(1)]],\n", f);
      fputs("    device const half *V [[buffer(2)]],\n", f);
      fputs("    device half *O [[buffer(3)]],\n", f);
      fputs("    constant BwppAttentionParams &p [[buffer(4)]],\n", f);
      fputs("    uint gid [[thread_position_in_grid]]) {\n", f);
      fputs("  if (gid >= p.M * p.N) { return; }\n", f);
      fputs("  O[gid] = half(0.0f);\n", f);
      fputs("}\n", f);
    }
  }
  if (has_softmax) {
    fputs("\nstruct BwppSoftmaxParams {\n", f);
    fputs("  uint rows;\n", f);
    fputs("  uint cols;\n", f);
    fputs("  uint ld;\n", f);
    fputs("};\n\n", f);
    fputs("kernel void bwpp_softmax_f16(\n", f);
    fputs("    device const half *X [[buffer(0)]],\n", f);
    fputs("    device half *Y [[buffer(1)]],\n", f);
    fputs("    constant BwppSoftmaxParams &p [[buffer(2)]],\n", f);
    fputs("    uint gid [[thread_position_in_grid]]) {\n", f);
    fputs("  uint row = gid;\n", f);
    fputs("  if (row >= p.rows) { return; }\n", f);
    fputs("  float maxv = -INFINITY;\n", f);
    fputs("  for (uint c = 0; c < p.cols; ++c) {\n", f);
    fputs("    float v = float(X[row * p.ld + c]);\n", f);
    fputs("    maxv = max(maxv, v);\n", f);
    fputs("  }\n", f);
    fputs("  float sum = 0.0f;\n", f);
    fputs("  for (uint c = 0; c < p.cols; ++c) {\n", f);
    fputs("    float e = exp(float(X[row * p.ld + c]) - maxv);\n", f);
    fputs("    Y[row * p.ld + c] = half(e);\n", f);
    fputs("    sum += e;\n", f);
    fputs("  }\n", f);
    fputs("  float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;\n", f);
    fputs("  for (uint c = 0; c < p.cols; ++c) {\n", f);
    fputs("    Y[row * p.ld + c] = half(float(Y[row * p.ld + c]) * inv);\n", f);
    fputs("  }\n", f);
    fputs("}\n", f);
  }
  if (has_rmsnorm) {
    fputs("\nstruct BwppRmsnormParams {\n", f);
    fputs("  uint rows;\n", f);
    fputs("  uint cols;\n", f);
    fputs("  uint ld;\n", f);
    fputs("  float eps;\n", f);
    fputs("};\n\n", f);
    fputs("kernel void bwpp_rmsnorm_f16(\n", f);
    fputs("    device const half *X [[buffer(0)]],\n", f);
    fputs("    device const half *Gamma [[buffer(1)]],\n", f);
    fputs("    device half *Y [[buffer(2)]],\n", f);
    fputs("    constant BwppRmsnormParams &p [[buffer(3)]],\n", f);
    fputs("    device const half *Beta [[buffer(4)]],\n", f);
    fputs("    uint gid [[thread_position_in_grid]]) {\n", f);
    fputs("  uint row = gid;\n", f);
    fputs("  if (row >= p.rows) { return; }\n", f);
    fputs("  float sumsq = 0.0f;\n", f);
    fputs("  for (uint c = 0; c < p.cols; ++c) {\n", f);
    fputs("    float v = float(X[row * p.ld + c]);\n", f);
    fputs("    sumsq += v * v;\n", f);
    fputs("  }\n", f);
    fputs("  float inv = rsqrt(sumsq / float(p.cols) + p.eps);\n", f);
    fputs("  for (uint c = 0; c < p.cols; ++c) {\n", f);
    fputs("    float v = float(X[row * p.ld + c]) * inv;\n", f);
    fputs("    float g = Gamma ? float(Gamma[c]) : 1.0f;\n", f);
    fputs("    float b = Beta ? float(Beta[c]) : 0.0f;\n", f);
    fputs("    Y[row * p.ld + c] = half(v * g + b);\n", f);
    fputs("  }\n", f);
    fputs("}\n", f);
  }
  fclose(f);
  bwpp_tile_kernel_destroy(tile);
  return BWPP_OK;
}

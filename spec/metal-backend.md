# Metal Backend v0.1

## Pipeline
1. IR graph fusion (matmul + bias + activation).
2. Tile selection and vectorization (Metal SIMD + threadgroup memory).
3. Emit MSL kernels + metadata for dispatch.

## Kernel constraints
- Static shapes only.
- Explicit layout and stride.
- No dynamic allocation inside kernels.

## PIM-inspired principles
- Explicit data movement between global and threadgroup memory.
- Scratchpad-first tiling discipline for reuse.
- SPMD execution model mapped to Metal threadgroups and SIMD lanes.

## Dispatch model
- Host runtime builds command buffers.
- Each fused region becomes one kernel dispatch.
- Buffer reuse handled by memory planner.

## Metadata
- Each emitted kernel includes metadata for op counts and reversible regions.
- Reversible policy is recorded (`store`, `recompute`, `auto`).
- `fused_attention_candidate=1` is appended when the graph matches the
  QK^T → softmax → V pattern (experimental).
- `attention_plan=tile_ir_stub` marks a Tile-IR-level fused attention plan
  placeholder.
- `bwpp.plan` lines enumerate the tile-op sequence for fused attention.

## Device profiles
- GPU-first targeting Apple Silicon (M4-class default).
- Tile sizes and vector widths selected per device profile.
- Feature probing via Metal device capabilities.

## Targets
- MSL 2.0+.

## Notes
- v0.1 focuses on matmul, add, softmax, rmsnorm, and a naive fused attention kernel.
- v0.2 adds layernorm, GELU, reduce, transpose.

## Attention kernel (v0.1)
- Kernel name: `bwpp_attention_f16`
- Buffers:
  - 0: Q (f16, M x K)
  - 1: K (f16, N x K)
  - 2: V (f16, N x D)
  - 3: O (f16, M x D)
  - 4: params (struct below)
- Params: `{ M, N, K, D, ldq, ldk, ldv, ldo }`
- Dispatch: 2D threadgroups, `threadsPerThreadgroup=(tile,tile,1)` and
  `threadgroups=(ceil(D/tile), ceil(M/tile), 1)`.

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

## Device profiles
- GPU-first targeting Apple Silicon (M4-class default).
- Tile sizes and vector widths selected per device profile.
- Feature probing via Metal device capabilities.

## Targets
- MSL 2.0+.

## Notes
- v0.1 focuses on matmul, add, softmax, rmsnorm.
- v0.2 adds layernorm, GELU, reduce, transpose.

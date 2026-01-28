# Device Profiles

BW++ is GPU-first and targets Metal on Apple Silicon. The primary profile is Apple M4-class GPUs. A CPU backend is optional and designed to emulate a GPU-like execution model (SPMD over SIMD + threads).

## GPU profile (Metal)
- Static shapes and layouts enable aggressive fusion and tiling.
- Kernels use threadgroup memory for reuse and bandwidth reduction.
- Dispatch favors large, fused kernels to amortize overhead.
- Default tuning targets M4-class GPUs; other Apple GPUs use fallback profiles.
- SPMD-style mapping: threadgroup tiles + SIMD lanes.

## CPU profile (optional, later)
- SPMD execution model mapped to SIMD lanes.
- Tiled loops for cache locality.
- Workgroup-style threading for parallel blocks.

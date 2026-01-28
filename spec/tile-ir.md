# Tile IR v0.1

Tile IR is a low-level kernel representation for Metal codegen.
It captures tile sizes, memory placement, and op sequencing.

## Concepts
- **Tile shape**: (m, n, k) sizes for matmul tiles.
- **Memory space**: global, threadgroup, register.
- **Block shape**: threadgroup-level tile dimensions.
- **SPMD mapping**: tile programs map to threadgroup + SIMD lanes.
- **Roles**: A/B/C for load/store association.

## Core ops (v0.1)
- `matmul` tile op
- `load` (global -> threadgroup)
- `store` (register -> global)
- `elementwise` (for fused epilogues such as add/silu)

## Example (conceptual)
- block: (128, 128, 32)
- op: matmul tile
  - A: threadgroup
  - B: threadgroup
  - C: register

## Lowering
- Graph IR -> fused regions -> Tile IR -> MSL kernel.

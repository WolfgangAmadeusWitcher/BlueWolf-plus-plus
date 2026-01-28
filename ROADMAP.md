# BW++ Roadmap

## v0.1 (Metal-first MVP)
- Core syntax and type system for tensor graphs
- Static shapes and layouts
- IR for tensor ops
- Metal backend for core LLM ops: matmul, add, softmax, rmsnorm
- Graph-level autodiff for a small op set (forward + backward)
- Simple memory planner and buffer reuse
- Minimal runtime (arena + tensor metadata)
- Benchmarks for matmul and attention kernels
- Experimental track: tile-DSL kernel IR and fused attention prototype

## v0.2
- More ops: layernorm, GELU, reduce, transpose, batch-matmul, swiglu
- Kernel fusion and auto-tiling
- Debug tooling (IR dump, kernel inspection)
- Experimental track: auto-scheduler and cost-model tuning

## v0.3
- Training-ready LLM primitives
- Optional CPU reference backend for correctness and verification

## v1.0
- Stable DSL and IR
- Reproducible benchmarks and model zoo
- Community governance and contribution workflow

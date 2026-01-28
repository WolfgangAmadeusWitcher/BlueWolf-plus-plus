# BW++

Metal-first tensor DSL for LLM training and dense linear algebra.

BW++ is a focused, high-performance DSL that targets Metal GPUs on Apple Silicon. It is not a general-purpose language. The goal is to deliver extreme performance and minimal memory overhead for dense tensor programs, especially LLM training and inference.

This project is an open-source initiative by SGN Tech Community.

## Vision
- Build the first language designed end-to-end for LLM training on Metal GPUs.
- Provide a tiny tensor DSL that compiles to fused Metal compute kernels.
- Make LLM-critical ops fast by default (matmul, attention, softmax, layernorm).
- Keep runtime overhead near-zero with static shapes and memory planning.

## Status
Pre-alpha. Specs and compiler skeleton only.

## Design goals
- Static shapes and layouts are required.
- AOT compilation to Metal (MSL) with aggressive fusion and tiling.
- Minimal runtime: arena allocator + tensor metadata only.
- Turing-complete at compile time via recursion in the meta layer.
- GPU-first device profiles (Metal). Primary profile: Apple M4-class GPU.
- CPU-as-GPU backend is optional and later.
- PIM-inspired principles: explicit data movement, scratchpad/tiling discipline, SPMD model.

## Non-goals
- General-purpose programming.
- Dynamic shapes at runtime.
- JIT or runtime code generation.
- Garbage collection or exceptions.

## Repository layout
- `spec/` language, IR, and backend specs
- `compiler/` C compiler skeleton
- `runtime/` minimal runtime, Metal artifacts, and CPU reference
- `examples/` DSL examples

## Roadmap
See `ROADMAP.md`.

## Tests
- Build compiler: `make -C compiler`
- CPU ref tests: `make -C runtime/cpu` then `./runtime/cpu/bwpp_cpu_test` and `./runtime/cpu/bwpp_cpu_norm_test`
- CPU Metal-parity tests (generate `.metal` from examples and validate via CPU ref):
  `make -C runtime/cpu cpu-metal-tests`

## License
Apache-2.0.

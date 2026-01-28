# BW++ Spec Overview

BW++ is a tensor DSL focused on dense linear algebra and LLM workloads. Programs describe tensor graphs with static shapes and layouts. The compiler performs fusion, tiling, and memory planning, then emits Metal kernels.

Key principles:
- Static shapes, explicit layout, no dynamic dispatch.
- Compile-time graph construction via a minimal meta layer.
- AOT to Metal Shading Language (MSL).
- Minimal runtime: arena allocator + tensor metadata only.
- PIM-inspired discipline: explicit data movement, scratchpad tiling, SPMD model.

Docs:
- `language.md`: surface syntax and types
- `ir.md`: intermediate representation
- `tile-ir.md`: tile-level kernel IR
- `metal-backend.md`: kernel generation and codegen
- `runtime.md`: runtime and memory model
- `benchmarks.md`: performance targets and methodology
- `llm-primitives.md`: LLM-specific op set and fusion targets
- `device-profiles.md`: hardware targets and scheduling assumptions
- `experimental-track.md`: cutting-edge compiler techniques and priorities
- `autodiff.md`: reverse-mode rules and reversible policies

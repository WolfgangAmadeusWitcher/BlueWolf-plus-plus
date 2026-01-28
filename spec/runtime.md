# Runtime v0.1

## Memory model
- Arena allocator for tensor buffers.
- All tensor buffers are owned by the runtime, not the language.
- Buffer reuse based on liveness analysis.

## Tensor metadata
- `dtype`, `shape`, `layout`, `stride`, `buffer_id`.

## Host API (sketch)
- `bwpp_context_create()`
- `bwpp_tensor_create(ctx, dtype, shape, layout)`
- `bwpp_run(graph, inputs, outputs)`

## CPU reference backend (validation)
- `runtime/cpu/` provides a tiny float32 reference for matmul and fused epilogues.
- Used for correctness checks without requiring Metal hardware.

# BW++ Language v0.1

## Goals
- Describe tensor graphs for dense linear algebra and LLM primitives.
- Keep the surface syntax tiny and strict.
- Support compile-time recursion in a meta layer to build graphs.

## Core types
- `tensor<dtype, shape, layout>`
  - `dtype`: `f16`, `bf16`, `f32`
  - `shape`: static sizes, e.g. `[B, M, N]`
  - `layout`: `row_major`, `col_major`, or blocked variants
- `int`, `bool`

## Values
- Scalars are immediate constants.
- Tensors are immutable by default.

## Functions
- Functions are pure unless marked `@impure`.
- Recursion is allowed.
- No loops in v0.1 (`while`/`for` are rejected).
- Tail-recursion is optimized into loops by the compiler.

## Built-in ops (v0.1)
- `@` matmul
- `batch_matmul`
- `transpose`, `permute`, `reshape`
- `add`, `sub`, `mul`, `div` (broadcasting)
- `reduce_sum`, `reduce_max` (axis)
- `softmax` (axis)
- `rmsnorm` (axis, epsilon; optional beta)
- `silu`

Notes:
- `add(x, bias)` enables matmul+bias fusion in the compiler.
- `add(x, reshape(bias, [N]))` and `add(x, permute(bias, [0]))` are accepted
  forms for bias when shapes are compatible.

## Reversible regions (experimental)
- `@reversible` blocks mark subgraphs for recompute-friendly backward passes.
- The compiler may drop saved activations inside these regions and recompute
  them during backprop, trading compute for memory.
- Not all ops are reversible; the compiler will warn on unsupported ops.

## Syntax (sketch)

fn matmul(a: tensor<f16,[M,K],row_major>, b: tensor<f16,[K,N],row_major>)
  -> tensor<f16,[M,N],row_major> {
  return a @ b
}

fn mlp(x: tensor<f16,[B,D],row_major>, w: tensor<f16,[D,H],row_major>)
  -> tensor<f16,[B,H],row_major> {
  return silu(x @ w)
}

@reversible
fn rev_block(x: tensor<f16,[B,D],row_major>, w: tensor<f16,[D,H],row_major>)
  -> tensor<f16,[B,H],row_major> {
  return silu(x @ w)
}

## Meta layer
- `@meta` functions execute at compile time to construct graphs.
- Meta functions may use recursion to generate repeated blocks.
- Meta functions cannot access runtime tensor data.

## Disallowed features
- Dynamic shapes
- Exceptions
- GC
- Runtime reflection

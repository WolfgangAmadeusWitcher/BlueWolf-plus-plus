# LLM Primitives v0.1 (Locked)

This op set is the minimum required to express and train transformer-style LLMs
while keeping the compiler and runtime narrow and highly optimized.

## Core forward ops
- `matmul` (2D)
- `batch_matmul` (3D/4D)
- `transpose` / `permute`
- `reshape` / `view`
- `add`, `sub`, `mul`, `div` (broadcasting)
- `reduce_sum`, `reduce_max` (axis)
- `softmax` (axis)
- `rmsnorm` (axis, epsilon)
- `silu` (activation)

## Fused patterns (first-class in codegen)
- `matmul + bias`
- `matmul + bias + silu`
- `swiglu` (fused `silu(x) * y`)
- `attention` block (qk^t + softmax + v)
- `residual + norm`

## Graph-level autodiff
- Reverse-mode autodiff on the v0.1 op set.
- Backward kernels are generated or fused when profitable.

## Attention (decomposed)
- `scores = q @ transpose(k)`
- `probs = softmax(scores)`
- `out = probs @ v`

## Constraints
- Shapes are static.
- Layout is explicit.
- Ops are fused when possible.

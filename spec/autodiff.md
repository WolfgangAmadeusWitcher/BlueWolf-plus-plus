# Autodiff v0.1

BW++ uses graph-level reverse-mode autodiff on the v0.1 op set.
Gradients are produced as a new graph that shares the same static shapes.

## Reversible policy
- Regions marked `@reversible` prefer recompute over storing activations.
- The policy can be overridden per region: `store`, `recompute`, `auto`.
- Unsupported ops inside a reversible region trigger a compile-time warning
  and fall back to `store` for that region.

## Supported ops (v0.1)
- `matmul`, `batch_matmul`
- `transpose`, `permute`, `reshape`
- `add`, `sub`, `mul`, `div`
- `reduce_sum`, `reduce_max`
- `softmax`
- `rmsnorm`
- `silu`

## Backward rules (high-level)
- `add/sub`: pass-through gradients with broadcasting reduction.
- `mul/div`: product/quotient rule with broadcasting reduction.
- `matmul`: standard GEMM gradients (dA = dY @ B^T, dB = A^T @ dY).
- `batch_matmul`: same as matmul per batch.
- `transpose/permute`: gradient is inverse permutation.
- `reshape`: gradient reshapes back to input shape.
- `reduce_sum`: expand gradient to input shape.
- `reduce_max`: requires argmax mask; in reversible regions recompute inputs.
- `softmax`: uses Jacobian-vector product; in reversible regions recompute.
- `rmsnorm`: requires input stats; recompute if reversible.
- `silu`: uses sigmoid(x); recompute if reversible.

## Save vs recompute policy
Default behavior is to store necessary intermediates unless a reversible
region is active. The scheduler uses a cost model to decide recompute in
`auto` mode.

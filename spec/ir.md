# BW++ IR v0.1

The IR is a static tensor graph:
- Nodes are ops with typed, shaped tensor inputs/outputs.
- Shapes and layouts are fully known at compile time.
- No control-flow in the runtime graph; branching is resolved in the meta layer.

## Supported ops (v0.1)
- `matmul`, `batch_matmul`
- `transpose`, `permute`, `reshape`
- `add`, `sub`, `mul`, `div`
- `reduce_sum`, `reduce_max`
- `softmax`
- `rmsnorm`
- `silu`

## Reversible regions (experimental)
- Nodes may belong to a reversible region.
- The scheduler may drop saved activations in reversible regions and
  recompute them during backprop.
- Each node has an optional `region_id` referencing a region table.

## Node fields
- `op`: e.g. `matmul`, `add`, `silu`, `softmax`
- `inputs`: tensor ids
- `attrs`: op-specific attributes (tiling hints, layout)
- `dtype`, `shape`, `layout`
- `region_id` (optional)
- `flags` (optional), e.g. `has_bias` for fused epilogues

Common attrs:
- `axis` for reductions, softmax, norm
- `epsilon` for rmsnorm
- `perm` for permute
- `shape` for reshape

## Region table
- `id`: integer
- `kind`: `normal` | `reversible`
- `policy`: `store` | `recompute` | `auto`

## Example

node0 = matmul(t0, t1) : tensor<f16,[M,N],row_major>
node1 = add(node0, bias) : tensor<f16,[M,N],row_major>
node2 = silu(node1) : tensor<f16,[M,N],row_major>

## Lowering
- Graph -> fused regions -> kernel IR -> MSL source

## Graph dumps
`bwppc` can emit a DOT graph of the forward IR and autodiff IR:
`bwppc input.bwpp out.metal --dot out.dot --grad-dot out_grad.dot`

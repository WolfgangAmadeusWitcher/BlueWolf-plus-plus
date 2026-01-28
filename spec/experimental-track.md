# Experimental Track

BW++ uses cutting-edge compiler ideas to reach extreme performance on Metal GPUs.

## v0.1 priorities
1) Tile-DSL kernel IR and Metal codegen (Triton-style kernels)
2) Fused attention prototype (FlashAttention-style IO-aware kernels)
3) Reversible AD regions (memory-recompute tradeoff for training)
4) CPU-as-GPU fallback design (SPMD-on-SIMD model)

## v0.2 priorities
1) Auto-scheduler with search space + cost model (Ansor-style)
2) Autotuning infrastructure and kernel database
3) Broader fusion patterns (MLP blocks, residual + norm)
4) Reversible scheduling heuristics and cost model integration

## Notes
- These items are experimental and may change as benchmarks evolve.
- Each feature must ship with reproducible benchmark cases.

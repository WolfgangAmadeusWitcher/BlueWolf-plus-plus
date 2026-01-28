# Benchmarks v0.1

## Goals
- Show extreme speedups in dense tensor workloads.
- Track memory peak and bandwidth utilization.

## Baselines
- Objective-C baseline (same algorithms)
- Naive C implementation (no vectorization)

## Targets
- matmul (LLM-sized shapes)
- attention block (QK^T, softmax, V)
- rmsnorm
- mlp block (matmul + swiglu + matmul)

## Metrics
- throughput (GFLOPS)
- peak memory
- end-to-end latency

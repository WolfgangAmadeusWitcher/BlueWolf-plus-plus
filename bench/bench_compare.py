#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time


def run_bwpp(bench_path, args):
    if not os.path.exists(bench_path):
        print("bwpp_bench not found. Build it with: make -C bench")
        return
    cmd = [bench_path]
    if args.iters is not None:
        cmd += ["--iters", str(args.iters)]
    if args.m is not None:
        cmd += ["--m", str(args.m)]
    if args.n is not None:
        cmd += ["--n", str(args.n)]
    if args.k is not None:
        cmd += ["--k", str(args.k)]
    if args.metal is not None:
        cmd += ["--metal", args.metal]
    print("== BW++ CPU bench ==")
    subprocess.run(cmd, check=False)


def run_mlx(args):
    try:
        import mlx.core as mx
    except Exception as exc:
        print(f"MLX not available: {exc}")
        print("Install MLX to run Metal baseline: pip install mlx")
        return

    m = args.m or 256
    n = args.n or 256
    k = args.k or 256
    iters = args.iters or 10
    rows = args.rows or 256
    cols = args.cols or 256

    print("== MLX Metal baseline ==")
    a = mx.random.uniform(shape=(m, k), dtype=mx.float16)
    b = mx.random.uniform(shape=(k, n), dtype=mx.float16)
    mx.eval(a, b)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
        mx.eval(c)
    t1 = time.perf_counter()
    secs = t1 - t0
    flops = 2.0 * m * n * k * iters
    gflops = (flops / 1e9) / secs if secs > 0 else 0.0
    print(f"matmul: M={m} N={n} K={k} iters={iters} time={secs:.6f}s gflops={gflops:.2f}")

    x = mx.random.uniform(shape=(rows, cols), dtype=mx.float16)
    mx.eval(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        y = mx.softmax(x, axis=-1)
        mx.eval(y)
    t1 = time.perf_counter()
    print(f"softmax: rows={rows} cols={cols} iters={iters} time={t1 - t0:.6f}s")

    gamma = mx.ones((cols,), dtype=mx.float16)
    eps = 1e-5
    t0 = time.perf_counter()
    for _ in range(iters):
        mean = mx.mean(x * x, axis=1, keepdims=True)
        inv = mx.rsqrt(mean + eps)
        y = x * inv * gamma
        mx.eval(y)
    t1 = time.perf_counter()
    print(f"rmsnorm: rows={rows} cols={cols} iters={iters} time={t1 - t0:.6f}s")


def main():
    parser = argparse.ArgumentParser(description="BW++ vs MLX baseline")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--metal", type=str, default=None)
    parser.add_argument("--skip-bwpp", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    args = parser.parse_args()

    bench_path = os.path.join(os.path.dirname(__file__), "bwpp_bench")
    if not args.skip_bwpp:
        run_bwpp(bench_path, args)
    if not args.skip_mlx:
        run_mlx(args)


if __name__ == "__main__":
    sys.exit(main())

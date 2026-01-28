#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile


def run_cmd(cmd):
    return subprocess.run(cmd, check=False)


def run_bench(bench_path, args):
    if not os.path.exists(bench_path):
        print("bwpp_bench not found. Build it with: make -C bench")
        return None

    with tempfile.NamedTemporaryFile(prefix="bwpp_bench_", suffix=".json", delete=False) as tf:
        json_path = tf.name

    cmd = [bench_path, "--json", json_path]
    cmd += ["--iters", str(args.iters)]
    cmd += ["--m", str(args.m), "--n", str(args.n), "--k", str(args.k)]
    cmd += ["--rows", str(args.rows), "--cols", str(args.cols)]
    if args.metal:
        cmd += ["--metal", args.metal]

    run_cmd(cmd)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"failed to read bench json: {exc}")
        data = None
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass

    return data


def compare_metric(name, base, cur, tol):
    if base <= 0.0:
        return True, 0.0
    ratio = cur / base
    ok = ratio <= (1.0 + tol)
    return ok, ratio


def main():
    parser = argparse.ArgumentParser(description="BW++ CPU benchmark regression guard")
    parser.add_argument("--baseline", default="bench/baseline_cpu.json")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--tol", type=float, default=0.20, help="max slowdown ratio (e.g. 0.2 = 20%%)")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--metal", type=str, default=None)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    if not args.no_build:
        run_cmd(["make", "-C", "bench"])

    bench_path = os.path.join("bench", "bwpp_bench")
    data = run_bench(bench_path, args)
    if data is None:
        return 1

    baseline_path = args.baseline
    if args.update:
        os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"baseline updated: {baseline_path}")
        return 0

    if not os.path.exists(baseline_path):
        print(f"baseline missing: {baseline_path}")
        print("run with --update to create it")
        return 1

    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    checks = [
        ("matmul", "time_s"),
        ("softmax", "time_s"),
        ("rmsnorm", "time_s"),
    ]

    failed = 0
    for group, key in checks:
        if group not in base or group not in data:
            print(f"missing metric: {group}.{key}")
            failed += 1
            continue
        b = float(base[group].get(key, 0.0))
        c = float(data[group].get(key, 0.0))
        ok, ratio = compare_metric(group, b, c, args.tol)
        status = "OK" if ok else "SLOW"
        print(f"{group}.{key}: base={b:.6f}s cur={c:.6f}s ratio={ratio:.2f} [{status}]")
        if not ok:
            failed += 1

    if failed:
        print(f"regression detected: {failed} metric(s) exceeded tolerance")
        return 1

    print("regression check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

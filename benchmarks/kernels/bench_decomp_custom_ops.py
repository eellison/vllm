# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark vLLM CustomOp CUDA kernels vs Inductor-compiled decompositions.

Section 1 — Individual ops:
    CustomOp.forward_cuda  vs  torch.compile(CustomOp.forward_native)

Section 2 — Fused patterns:
    Single fused CUDA kernel (where available) or 2 sequential CUDA kernels
    vs  torch.compile(composed forward_native calls)

Section 3 — INT8 inline_asm ablation (--asm-ablation):
    Decomp speedup vs CUDA with and without PTX cvt.rni.sat.s8.f32

Usage:
    python benchmarks/kernels/bench_decomp_custom_ops.py
    python benchmarks/kernels/bench_decomp_custom_ops.py --sweep
    python benchmarks/kernels/bench_decomp_custom_ops.py --sweep --csv results.csv
    python benchmarks/kernels/bench_decomp_custom_ops.py --asm-ablation
    python benchmarks/kernels/bench_decomp_custom_ops.py --verify
"""

import argparse
import csv
import io
import math
from typing import Any

import torch
import torch._dynamo

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.input_quant_int8 import QuantInt8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

FP8_DTYPE = current_platform.fp8_dtype()

SWEEP_SHAPES = [
    (1, 4096), (4, 4096), (32, 4096), (128, 4096),
    (512, 4096), (2048, 4096), (512, 8192), (512, 14336),
]


# ============================================================================
# Helpers
# ============================================================================

def cudagraph_wrap(fn, inputs):
    """Capture fn(*inputs) into a CUDA graph and return a replay callable."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(*inputs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(*inputs)
    return lambda: g.replay()


def bench(fn, inputs, warmup=10, rep=100):
    return triton.testing.do_bench(cudagraph_wrap(fn, inputs),
                                   warmup=warmup, rep=rep)


def compile_and_bench(fn, inputs, args):
    """Compile fn, run multiple trials, return best time."""
    best = float("inf")
    for _ in range(args.trials):
        torch._dynamo.reset()
        compiled = torch.compile(fn)
        for _ in range(3):
            compiled(*inputs)
        best = min(best, bench(compiled, inputs, args.warmup, args.rep))
    return best


def verify(name, cuda_fn, native_fn, inputs, atol=0.05, rtol=0.05):
    a = cuda_fn(*inputs)
    compiled = torch.compile(native_fn)
    for _ in range(3):
        compiled(*inputs)
    b = compiled(*inputs)
    if not isinstance(a, tuple):
        a, b = (a,), (b,)
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra is None or rb is None:
            continue
        ra_f = ra.float() if ra.is_floating_point() else ra.to(torch.float32)
        rb_f = rb.float() if rb.is_floating_point() else rb.to(torch.float32)
        if not torch.allclose(ra_f, rb_f, atol=atol, rtol=rtol):
            diff = (ra_f - rb_f).abs().max().item()
            print(f"  WARN [{name}] output[{i}] max_diff={diff:.4f}")
            return False
    return True


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def make_op(op, method="cuda"):
    """Return a closure calling op.forward_cuda or op.forward_native."""
    fn = op.forward_cuda if method == "cuda" else op.forward_native
    return lambda *args: fn(*args)


# ============================================================================
# Op definitions (data-driven)
# ============================================================================

def build_individual_ops(M, N, dtype, device, scale):
    """Return list of (name, cuda_fn, native_fn, input_fn)."""
    ops = []

    def add(name, op, input_fn):
        ops.append((name, make_op(op, "cuda"), make_op(op, "native"), input_fn))

    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    add("rms_norm", norm,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),))

    norm2 = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    add("fused_add_rms_norm", norm2,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),
                 torch.randn(M, N, dtype=dtype, device=device)))

    add("static_fp8_quant",
        QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR),
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale))

    add("dynamic_fp8_quant_per_tensor",
        QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR),
        lambda: (torch.randn(M, N, dtype=dtype, device=device),))

    add("dynamic_fp8_quant_per_token",
        QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN),
        lambda: (torch.randn(M, N, dtype=dtype, device=device),))

    add("silu_and_mul", SiluAndMul(),
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device),))

    add("static_int8_quant", QuantInt8(static=True, symmetric=True),
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale))

    add("dynamic_int8_quant", QuantInt8(static=False, symmetric=True),
        lambda: (torch.randn(M, N, dtype=dtype, device=device),))

    return ops


def build_fused_ops(M, N, dtype, device, scale):
    """Return list of (name, cuda_fn, native_fn, input_fn)."""
    ops = []

    def add_fused(name, ops_list, fused_cuda_op=None):
        """Build fused cuda/native fns from a list of (op, extra_args) pairs.

        If fused_cuda_op is provided, use it for CUDA; otherwise chain
        forward_cuda calls sequentially (2 kernel launches).
        """
        def native_fn(*args):
            result = args
            for op, _ in ops_list:
                if not isinstance(result, tuple):
                    result = (result,)
                result = op.forward_native(*result)
            return result

        if fused_cuda_op is not None:
            cuda_fn = fused_cuda_op
        else:
            def cuda_fn(*args):
                result = args
                for op, _ in ops_list:
                    if not isinstance(result, tuple):
                        result = (result,)
                    result = op.forward_cuda(*result)
                return result

        ops.append((name, cuda_fn, native_fn))

    # -- RMSNorm + static FP8 (fused CUDA kernel) --
    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)

    def rms_fp8_cuda(x, s, _n=norm):
        out = torch.empty_like(x, dtype=FP8_DTYPE)
        torch.ops._C.rms_norm_static_fp8_quant(
            out, x, _n.weight, s, _n.variance_epsilon)
        return out, s

    def rms_fp8_native(x, s, _n=norm,
                       _q=QuantFP8(static=True,
                                   group_shape=GroupShape.PER_TENSOR)):
        return _q.forward_native(_n.forward_native(x), s)

    ops.append(("rms_norm_static_fp8_quant", rms_fp8_cuda, rms_fp8_native,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale)))

    # -- FusedAdd+RMSNorm + static FP8 (fused CUDA kernel) --
    norm2 = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    quant_fp8_2 = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    def fadd_fp8_cuda(x, res, s, _n=norm2):
        out = torch.empty_like(x, dtype=FP8_DTYPE)
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out, x, res, _n.weight, s, _n.variance_epsilon)
        return out, res

    def fadd_fp8_native(x, res, s, _n=norm2, _q=quant_fp8_2):
        normed, r = _n.forward_native(x, res)
        q, _ = _q.forward_native(normed, s)
        return q, r

    ops.append(("fused_add_rms_norm_static_fp8", fadd_fp8_cuda,
        fadd_fp8_native,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),
                 torch.randn(M, N, dtype=dtype, device=device), scale)))

    # -- RMSNorm + dynamic per-token FP8 (fused CUDA kernel) --
    norm3 = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)

    def rms_dyn_fp8_cuda(x, _n=norm3):
        out = torch.empty_like(x, dtype=FP8_DTYPE)
        scales = torch.empty((x.shape[0], 1), device=x.device,
                             dtype=torch.float32)
        torch.ops._C.rms_norm_dynamic_per_token_quant(
            out, x, _n.weight, scales, _n.variance_epsilon, None, None)
        return out, scales

    def rms_dyn_fp8_native(x, _n=norm3,
                           _q=QuantFP8(static=False,
                                       group_shape=GroupShape.PER_TOKEN)):
        return _q.forward_native(_n.forward_native(x))

    ops.append(("rms_norm_dyn_per_token_fp8", rms_dyn_fp8_cuda,
        rms_dyn_fp8_native,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),)))


    return ops


# ============================================================================
# Main benchmark loop
# ============================================================================

def run_section(section_name, ops_with_inputs, args):
    """Benchmark a list of ops, print results, return CSV rows."""
    print(f"\n--- {section_name} ---")
    print(f"{'Op':<40} {'CUDA(ms)':>10} {'Decomp(ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    rows = []
    for name, cuda_fn, native_fn, input_fn in ops_with_inputs:
        torch._dynamo.reset()
        inputs = input_fn()

        if args.verify:
            verify(name, cuda_fn, native_fn, inputs)

        cuda_ms = bench(cuda_fn, inputs, args.warmup, args.rep)

        if args.no_compile:
            native_ms = bench(native_fn, inputs, args.warmup, args.rep)
        else:
            native_ms = compile_and_bench(native_fn, inputs, args)

        speedup = cuda_ms / native_ms
        marker = " << decomp wins" if speedup > 1.05 else ""
        print(f"{name:<40} {cuda_ms:>10.4f} {native_ms:>10.4f}"
              f" {speedup:>9.2f}x{marker}")
        rows.append({"section": section_name, "op": name,
                      "cuda_ms": f"{cuda_ms:.4f}",
                      "decomp_ms": f"{native_ms:.4f}",
                      "speedup": f"{speedup:.2f}"})

    speedups = [float(r["speedup"]) for r in rows]
    if speedups:
        print(f"{'GEOMEAN':<40} {'':>10} {'':>10}"
              f" {geomean(speedups):>9.2f}x")
    return rows


def _add_no_asm_variants(ops):
    """Duplicate INT8 ops with inline_asm disabled, appending '_no_asm'."""
    from vllm.model_executor.layers.quantization import input_quant_int8

    if not input_quant_int8._HAS_INLINE_ASM:
        return []

    extra = []
    for name, cuda_fn, native_fn, input_fn in ops:
        if "int8" not in name.lower():
            continue

        def make_no_asm_native(orig_native):
            def no_asm_native(*args, _fn=orig_native):
                input_quant_int8._HAS_INLINE_ASM = False
                try:
                    return _fn(*args)
                finally:
                    input_quant_int8._HAS_INLINE_ASM = True
            return no_asm_native

        extra.append((name + "_no_asm", cuda_fn,
                       make_no_asm_native(native_fn), input_fn))
    return extra


def bench_shape(M, N, args):
    """Benchmark all ops at a single (M, N) shape."""
    dtype, device = torch.bfloat16, "cuda"
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    print(f"\n{'='*80}\n  Shape: [{M}, {N}]\n{'='*80}")

    individual = build_individual_ops(M, N, dtype, device, scale)
    fused = build_fused_ops(M, N, dtype, device, scale)

    if args.asm_ablation:
        individual += _add_no_asm_variants(individual)
        fused += _add_no_asm_variants(fused)

    rows = []
    rows += run_section("Individual Ops", individual, args)
    rows += run_section("Fused Patterns", fused, args)

    shape_str = f"[{M},{N}]"
    for r in rows:
        r["shape"] = shape_str
    return rows


# ============================================================================
# CSV output
# ============================================================================

def write_csv(rows, csv_path, shapes):
    """Write pivoted CSV: one row per op, shapes as columns."""
    if not rows:
        return
    shape_strs = [f"[{m},{n}]" for m, n in shapes]

    from collections import OrderedDict
    grouped: dict[tuple[str, str], dict[str, dict]] = OrderedDict()
    for r in rows:
        key = (r["section"], r["op"])
        grouped.setdefault(key, {})[r.get("shape", "")] = r

    buf = io.StringIO()
    header = ["section", "op", "geomean_speedup"]
    for s in shape_strs:
        header.extend([f"{s}_speedup", f"{s}_cuda_ms", f"{s}_decomp_ms"])
    writer = csv.writer(buf)
    writer.writerow(header)

    for (section, op), by_shape in grouped.items():
        speedups = [float(by_shape[s]["speedup"])
                    for s in shape_strs
                    if s in by_shape and by_shape[s].get("speedup")]
        gm = geomean(speedups) if speedups else 0.0
        row = [section, op, f"{gm:.2f}"]
        for s in shape_strs:
            if s in by_shape:
                row.extend([by_shape[s].get("speedup", ""),
                            by_shape[s].get("cuda_ms", ""),
                            by_shape[s].get("decomp_ms", "")])
            else:
                row.extend(["", "", ""])
        writer.writerow(row)

    text = buf.getvalue()
    if csv_path:
        with open(csv_path, "w") as f:
            f.write(text)
        print(f"\nCSV written to {csv_path}")
    else:
        print(f"\n--- CSV ---\n{text}", end="")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep standard shapes")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Benchmark eager (no Inductor)")
    parser.add_argument("--asm-ablation", action="store_true",
                        help="Compare INT8 inline_asm vs fallback")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write CSV to file")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1,
                        help="Recompile N times, take best (filters noise)")
    args = parser.parse_args()

    print("vLLM CustomOp Decomposition Benchmark")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {'compiled' if not args.no_compile else 'eager'}")

    shapes = SWEEP_SHAPES if args.sweep else [(args.m, args.n)]
    print(f"Shapes: {shapes}")
    print("=" * 80)

    rows = []
    for M, N in shapes:
        rows += bench_shape(M, N, args)

    write_csv(rows, args.csv, shapes)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()

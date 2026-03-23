# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark vLLM CustomOp CUDA kernels vs Inductor-compiled decompositions.

Section 1 — Individual ops:
    CustomOp.forward_cuda  vs  torch.compile(CustomOp.forward_native)

Section 2 — Fused patterns:
    Single fused CUDA kernel (where available) or 2 sequential CUDA kernels
    vs  torch.compile(composed forward_native calls)

Section 3 — INT8 inline_asm ablation:
    inline_asm_elementwise (PTX cvt.rni.sat.s8.f32) vs pure-torch fallback
    (round+clamp+cast) on fused patterns to measure the impact of the
    single-instruction saturating cast.

Usage:
    python benchmarks/kernels/bench_decomp_custom_ops.py
    python benchmarks/kernels/bench_decomp_custom_ops.py --m 2048 --n 7168
    python benchmarks/kernels/bench_decomp_custom_ops.py --sweep
    python benchmarks/kernels/bench_decomp_custom_ops.py --sweep --csv results.csv
    python benchmarks/kernels/bench_decomp_custom_ops.py --verify
    python benchmarks/kernels/bench_decomp_custom_ops.py --no-compile
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


# ============================================================================
# Helpers
# ============================================================================

def cudagraph_wrap(fn, inputs):
    """Capture fn(*inputs) into a CUDA graph and return a replay callable."""
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(*inputs)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(*inputs)

    def replay():
        g.replay()

    return replay


def bench(fn, inputs, warmup=10, rep=100):
    replay = cudagraph_wrap(fn, inputs)
    return triton.testing.do_bench(replay, warmup=warmup, rep=rep)


def verify(name, fn_a, fn_b, inputs, atol=0.05, rtol=0.05):
    a = fn_a(*inputs)
    compiled_b = compile_fn(fn_b)
    warmup_compile(compiled_b, inputs)
    b = compiled_b(*inputs)
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


def compile_fn(fn):
    compiled = torch.compile(fn)
    return compiled


def warmup_compile(compiled_fn, inputs, n=3):
    for _ in range(n):
        compiled_fn(*inputs)


# ============================================================================
# Individual op wrappers
# ============================================================================

def make_rmsnorm(N, dtype, device):
    op = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)

    def cuda_fn(x):
        return op.forward_cuda(x)

    def native_fn(x):
        return op.forward_native(x)

    return cuda_fn, native_fn


def make_fused_add_rmsnorm(N, dtype, device):
    op = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)

    def cuda_fn(x, residual):
        return op.forward_cuda(x, residual)

    def native_fn(x, residual):
        return op.forward_native(x, residual)

    return cuda_fn, native_fn


def make_static_fp8_quant(device):
    op = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    def cuda_fn(x, scale):
        return op.forward_cuda(x, scale)

    def native_fn(x, scale):
        return op.forward_native(x, scale)

    return cuda_fn, native_fn


def make_dynamic_fp8_quant_per_tensor(device):
    op = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)

    def cuda_fn(x):
        return op.forward_cuda(x)

    def native_fn(x):
        return op.forward_native(x)

    return cuda_fn, native_fn


def make_dynamic_fp8_quant_per_token(device):
    op = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    def cuda_fn(x):
        return op.forward_cuda(x)

    def native_fn(x):
        return op.forward_native(x)

    return cuda_fn, native_fn


def make_silu_and_mul(device):
    op = SiluAndMul()

    def cuda_fn(x):
        return op.forward_cuda(x)

    def native_fn(x):
        return op.forward_native(x)

    return cuda_fn, native_fn


def make_static_int8_quant(device):
    op = QuantInt8(static=True, symmetric=True)

    def cuda_fn(x, scale):
        return op.forward_cuda(x, scale)

    def native_fn(x, scale):
        return op.forward_native(x, scale)

    return cuda_fn, native_fn


def make_dynamic_int8_quant(device):
    op = QuantInt8(static=False, symmetric=True)

    def cuda_fn(x):
        return op.forward_cuda(x)

    def native_fn(x):
        return op.forward_native(x)

    return cuda_fn, native_fn


# ============================================================================
# Fused pattern wrappers
# ============================================================================

def make_rmsnorm_static_fp8(N, dtype, device):
    """RMSNorm + static FP8 quant."""
    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    # Fused CUDA kernel
    has_fused = hasattr(torch.ops._C, "rms_norm_static_fp8_quant")

    def cuda_fn(x, scale):
        if has_fused:
            output = torch.empty_like(x, dtype=FP8_DTYPE)
            torch.ops._C.rms_norm_static_fp8_quant(
                output, x, norm.weight, scale, norm.variance_epsilon)
            return output, scale
        else:
            normed = norm.forward_cuda(x)
            return quant.forward_cuda(normed, scale)

    def native_fn(x, scale):
        normed = norm.forward_native(x)
        return quant.forward_native(normed, scale)

    label = "rms_norm_static_fp8_quant" if has_fused else "RMSNorm→FP8 (2 CUDA)"
    return cuda_fn, native_fn, label


def make_fused_add_rmsnorm_static_fp8(N, dtype, device):
    """FusedAdd+RMSNorm + static FP8 quant."""
    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    has_fused = hasattr(torch.ops._C,
                        "fused_add_rms_norm_static_fp8_quant")

    def cuda_fn(x, residual, scale):
        if has_fused:
            output = torch.empty_like(x, dtype=FP8_DTYPE)
            torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                output, x, residual, norm.weight, scale,
                norm.variance_epsilon)
            return output, residual
        else:
            normed, res_out = norm.forward_cuda(x, residual)
            q, _ = quant.forward_cuda(normed, scale)
            return q, res_out

    def native_fn(x, residual, scale):
        normed, res_out = norm.forward_native(x, residual)
        q, _ = quant.forward_native(normed, scale)
        return q, res_out

    label = ("fused_add_rms_norm_static_fp8" if has_fused
             else "FusedAddRMS→FP8 (2 CUDA)")
    return cuda_fn, native_fn, label


def make_rmsnorm_dynamic_per_token_fp8(N, dtype, device):
    """RMSNorm + dynamic per-token FP8 quant."""
    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    has_fused = hasattr(torch.ops._C, "rms_norm_dynamic_per_token_quant")

    def cuda_fn(x):
        if has_fused:
            output = torch.empty_like(x, dtype=FP8_DTYPE)
            scales = torch.empty((x.shape[0], 1), device=x.device,
                                 dtype=torch.float32)
            torch.ops._C.rms_norm_dynamic_per_token_quant(
                output, x, norm.weight, scales, norm.variance_epsilon,
                None, None)
            return output, scales
        else:
            normed = norm.forward_cuda(x)
            return quant.forward_cuda(normed)

    def native_fn(x):
        normed = norm.forward_native(x)
        return quant.forward_native(normed)

    label = ("rms_norm_dyn_per_token_fp8" if has_fused
             else "RMSNorm→dynFP8 (2 CUDA)")
    return cuda_fn, native_fn, label


def make_silu_and_mul_static_fp8(N, dtype, device):
    """SiluAndMul + static FP8 quant (no fused CUDA kernel exists)."""
    silu = SiluAndMul()
    quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    def cuda_fn(x, scale):
        activated = silu.forward_cuda(x)
        return quant.forward_cuda(activated, scale)

    def native_fn(x, scale):
        activated = silu.forward_native(x)
        return quant.forward_native(activated, scale)

    return cuda_fn, native_fn, "SiluAndMul→FP8 (2 CUDA)"


def make_rmsnorm_static_int8(N, dtype, device):
    """RMSNorm + static INT8 quant (no fused CUDA kernel exists)."""
    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    quant = QuantInt8(static=True, symmetric=True)

    def cuda_fn(x, scale):
        normed = norm.forward_cuda(x)
        return quant.forward_cuda(normed, scale)

    def native_fn(x, scale):
        normed = norm.forward_native(x)
        return quant.forward_native(normed, scale)

    return cuda_fn, native_fn, "RMSNorm→INT8 (2 CUDA)"


def make_silu_and_mul_static_int8(N, dtype, device):
    """SiluAndMul + static INT8 quant (no fused CUDA kernel exists)."""
    silu = SiluAndMul()
    quant = QuantInt8(static=True, symmetric=True)

    def cuda_fn(x, scale):
        activated = silu.forward_cuda(x)
        return quant.forward_cuda(activated, scale)

    def native_fn(x, scale):
        activated = silu.forward_native(x)
        return quant.forward_native(activated, scale)

    return cuda_fn, native_fn, "SiluAndMul→INT8 (2 CUDA)"


def make_rmsnorm_dynamic_int8(N, dtype, device):
    """RMSNorm + dynamic per-token INT8 quant."""
    norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
    quant = QuantInt8(static=False, symmetric=True)

    def cuda_fn(x):
        normed = norm.forward_cuda(x)
        return quant.forward_cuda(normed)

    def native_fn(x):
        normed = norm.forward_native(x)
        return quant.forward_native(normed)

    return cuda_fn, native_fn, "RMSNorm→dynINT8 (2 CUDA)"


# ============================================================================
# Main
# ============================================================================

SWEEP_SHAPES = [
    (1, 4096),
    (4, 4096),
    (32, 4096),
    (128, 4096),
    (512, 4096),
    (2048, 4096),
    (512, 8192),
    (512, 14336),
]


def bench_shape(M, N, args, gpu_name):
    """Benchmark all ops at a single (M, N) shape. Returns list of CSV rows."""
    dtype = torch.bfloat16
    device = "cuda"
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    csv_rows: list[dict[str, str]] = []
    shape_str = f"[{M},{N}]"

    print(f"\n{'='*80}")
    print(f"  Shape: [{M}, {N}]")
    print(f"{'='*80}")

    # ====================================================================
    # Section 1: Individual ops
    # ====================================================================
    print(f"\n--- Individual Ops ---")
    print(f"{'Op':<40} {'CUDA(ms)':>10} {'Decomp(ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    individual_ops: list[tuple[str, Any, Any, Any]] = []

    cuda_fn, native_fn = make_rmsnorm(N, dtype, device)
    individual_ops.append((
        "rms_norm", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    cuda_fn, native_fn = make_fused_add_rmsnorm(N, dtype, device)
    individual_ops.append((
        "fused_add_rms_norm", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),
                 torch.randn(M, N, dtype=dtype, device=device)),
    ))

    cuda_fn, native_fn = make_static_fp8_quant(device)
    individual_ops.append((
        "static_fp8_quant", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn = make_dynamic_fp8_quant_per_tensor(device)
    individual_ops.append((
        "dynamic_fp8_quant_per_tensor", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    cuda_fn, native_fn = make_dynamic_fp8_quant_per_token(device)
    individual_ops.append((
        "dynamic_fp8_quant_per_token", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    cuda_fn, native_fn = make_silu_and_mul(device)
    individual_ops.append((
        "silu_and_mul", cuda_fn, native_fn,
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device),),
    ))

    cuda_fn, native_fn = make_static_int8_quant(device)
    individual_ops.append((
        "static_int8_quant", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn = make_dynamic_int8_quant(device)
    individual_ops.append((
        "dynamic_int8_quant", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    for name, cuda_fn, native_fn, input_fn in individual_ops:
        torch._dynamo.reset()
        inputs = input_fn()
        if args.verify:
            verify(name, cuda_fn, native_fn, inputs)

        cuda_ms = bench(cuda_fn, inputs, args.warmup, args.rep)

        if args.no_compile:
            native_ms = bench(native_fn, inputs, args.warmup, args.rep)
        else:
            # Run multiple compile trials to filter autotune noise
            best_ms = float("inf")
            for _ in range(args.trials):
                torch._dynamo.reset()
                compiled = compile_fn(native_fn)
                warmup_compile(compiled, inputs)
                ms = bench(compiled, inputs, args.warmup, args.rep)
                best_ms = min(best_ms, ms)
            native_ms = best_ms

        speedup = cuda_ms / native_ms
        marker = " << decomp wins" if speedup > 1.05 else ""
        print(f"{name:<40} {cuda_ms:>10.4f} {native_ms:>10.4f}"
              f" {speedup:>9.2f}x{marker}")
        csv_rows.append({
            "section": "individual",
            "op": name,
            "cuda_ms": f"{cuda_ms:.4f}",
            "decomp_ms": f"{native_ms:.4f}",
            "speedup": f"{speedup:.2f}",
            "gpu": gpu_name,
            "shape": shape_str,
        })

    indiv_speedups = [
        float(r["speedup"]) for r in csv_rows
        if r["section"] == "individual" and r["shape"] == shape_str
    ]
    if indiv_speedups:
        gm = math.exp(sum(math.log(s) for s in indiv_speedups) / len(indiv_speedups))
        print(f"{'GEOMEAN':<40} {'':>10} {'':>10} {gm:>9.2f}x")

    # ====================================================================
    # Section 2: Fused patterns
    # ====================================================================
    print(f"\n--- Fused Patterns ---")
    print(f"{'Pattern':<40} {'CUDA(ms)':>10} {'Decomp(ms)':>10}"
          f" {'Speedup':>10}")
    print("-" * 80)

    fused_ops: list[tuple[str, Any, Any, Any]] = []

    cuda_fn, native_fn, label = make_rmsnorm_static_fp8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn, label = make_fused_add_rmsnorm_static_fp8(
        N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),
                 torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn, label = make_rmsnorm_dynamic_per_token_fp8(
        N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    cuda_fn, native_fn, label = make_silu_and_mul_static_fp8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn, label = make_rmsnorm_static_int8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn, label = make_silu_and_mul_static_int8(
        N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device), scale),
    ))

    cuda_fn, native_fn, label = make_rmsnorm_dynamic_int8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    for name, cuda_fn, native_fn, input_fn in fused_ops:
        torch._dynamo.reset()
        inputs = input_fn()

        cuda_ms = bench(cuda_fn, inputs, args.warmup, args.rep)

        if args.no_compile:
            native_ms = bench(native_fn, inputs, args.warmup, args.rep)
        else:
            best_ms = float("inf")
            for _ in range(args.trials):
                torch._dynamo.reset()
                compiled = compile_fn(native_fn)
                warmup_compile(compiled, inputs)
                ms = bench(compiled, inputs, args.warmup, args.rep)
                best_ms = min(best_ms, ms)
            native_ms = best_ms

        speedup = cuda_ms / native_ms
        marker = " << decomp wins" if speedup > 1.05 else ""
        print(f"{name:<40} {cuda_ms:>10.4f} {native_ms:>10.4f}"
              f" {speedup:>9.2f}x{marker}")
        csv_rows.append({
            "section": "fused",
            "op": name,
            "cuda_ms": f"{cuda_ms:.4f}",
            "decomp_ms": f"{native_ms:.4f}",
            "speedup": f"{speedup:.2f}",
            "gpu": gpu_name,
            "shape": shape_str,
        })

    fused_speedups = [
        float(r["speedup"]) for r in csv_rows
        if r["section"] == "fused" and r["shape"] == shape_str
    ]
    if fused_speedups:
        gm = math.exp(sum(math.log(s) for s in fused_speedups) / len(fused_speedups))
        print(f"{'GEOMEAN':<40} {'':>10} {'':>10} {gm:>9.2f}x")

    return csv_rows


def write_csv(csv_rows, csv_path, shapes):
    """Write pivoted CSV: one row per op, columns are shapes (speedup values).

    Also includes cuda_ms and decomp_ms for each shape so users can inspect
    absolute timings.
    """
    if not csv_rows:
        return

    # Collect unique shape strings in order
    shape_strs = [f"[{m},{n}]" for m, n in shapes]

    # Group rows by (section, op)
    from collections import OrderedDict
    grouped: dict[tuple[str, str], dict[str, dict]] = OrderedDict()
    for row in csv_rows:
        key = (row["section"], row["op"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][row["shape"]] = row

    # Build pivoted CSV
    buf = io.StringIO()
    # Header: section, op, geomean_speedup, then per-shape columns
    header = ["section", "op", "geomean_speedup"]
    for s in shape_strs:
        header.extend([f"{s}_speedup", f"{s}_cuda_ms", f"{s}_decomp_ms"])
    writer = csv.writer(buf)
    writer.writerow(header)

    for (section, op), by_shape in grouped.items():
        # Compute geomean of speedups across shapes
        speedups = []
        for s in shape_strs:
            if s in by_shape and by_shape[s]["speedup"]:
                speedups.append(float(by_shape[s]["speedup"]))
        geomean = (
            math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            if speedups else 0.0
        )

        row = [section, op, f"{geomean:.2f}"]
        for s in shape_strs:
            if s in by_shape:
                row.extend([
                    by_shape[s]["speedup"],
                    by_shape[s]["cuda_ms"],
                    by_shape[s]["decomp_ms"],
                ])
            else:
                row.extend(["", "", ""])
        writer.writerow(row)

    csv_text = buf.getvalue()

    if csv_path:
        with open(csv_path, "w") as f:
            f.write(csv_text)
        print(f"\nCSV written to {csv_path}")
    else:
        print(f"\n--- CSV ---")
        print(csv_text, end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep standard shapes: 1..2048 tokens, "
                             "4096..14336 hidden")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Benchmark eager decomposition (no Inductor)")
    parser.add_argument("--asm-ablation", action="store_true",
                        help="Compare INT8 inline_asm vs round+clamp fallback")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV file")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1,
                        help="Recompile N times and take best decomp time "
                             "(filters autotune non-determinism)")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name()

    print("vLLM CustomOp Decomposition Benchmark")
    print(f"Device: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {'compiled' if not args.no_compile else 'eager'}")

    shapes = SWEEP_SHAPES if args.sweep else [(args.m, args.n)]
    print(f"Shapes: {shapes}")
    print("=" * 80)

    csv_rows: list[dict[str, str]] = []

    for M, N in shapes:
        csv_rows.extend(bench_shape(M, N, args, gpu_name))

    if args.asm_ablation:
        ablation_rows = bench_int8_inline_asm(args, gpu_name)
        csv_rows.extend(ablation_rows)

    write_csv(csv_rows, args.csv, shapes)


def bench_int8_inline_asm(args, gpu_name):
    """Section 3: Show decomp speedup vs CUDA with and without inline_asm.

    Demonstrates that decomposition wins over CUDA kernels regardless of
    whether inline_asm_elementwise is available, and quantifies the extra
    benefit from inline_asm.
    """
    from vllm.model_executor.layers.quantization import input_quant_int8

    if not input_quant_int8._HAS_INLINE_ASM:
        print("\n--- Skipping inline_asm ablation "
              "(inline_asm_elementwise not available) ---")
        return []

    original_use_asm = input_quant_int8._USE_INLINE_ASM

    M, N = args.m, args.n
    dtype = torch.bfloat16
    device = "cuda"
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    print(f"\n--- INT8 inline_asm ablation: decomp speedup vs CUDA "
          f"with and without PTX cvt.rni.sat.s8.f32 ---")
    print(f"{'Pattern':<30} {'CUDA(ms)':>10} {'asm(ms)':>10}"
          f" {'fallback(ms)':>12} {'asm vs CUDA':>12}"
          f" {'fb vs CUDA':>12}")
    print("-" * 90)

    patterns: list[tuple[str, bool]] = [
        ("static_int8_quant", True),
        ("dynamic_int8_quant", False),
        ("RMSNorm+INT8", True),
        ("RMSNorm+dynINT8", False),
    ]

    csv_rows = []

    for name, is_static in patterns:
        torch._dynamo.reset()
        x = torch.randn(M, N, dtype=dtype, device=device)

        # CUDA baseline
        if name.startswith("RMSNorm"):
            norm_cuda = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
            quant_cuda = QuantInt8(static=is_static, symmetric=True)
            if is_static:
                def cuda_fn(x, s, _n=norm_cuda, _q=quant_cuda):
                    return _q.forward_cuda(_n.forward_cuda(x), s)
                inputs = (x, scale)
            else:
                def cuda_fn(x, _n=norm_cuda, _q=quant_cuda):
                    return _q.forward_cuda(_n.forward_cuda(x))
                inputs = (x,)
        else:
            quant_cuda = QuantInt8(static=is_static, symmetric=True)
            if is_static:
                def cuda_fn(x, s, _q=quant_cuda):
                    return _q.forward_cuda(x, s)
                inputs = (x, scale)
            else:
                def cuda_fn(x, _q=quant_cuda):
                    return _q.forward_cuda(x)
                inputs = (x,)

        cuda_ms = bench(cuda_fn, inputs, args.warmup, args.rep)

        # Decomp with and without asm
        results = {}
        for use_asm in [True, False]:
            torch._dynamo.reset()
            input_quant_int8._USE_INLINE_ASM = use_asm

            if name.startswith("RMSNorm"):
                norm = RMSNorm(hidden_size=N, eps=1e-6, dtype=dtype).to(device)
                quant = QuantInt8(static=is_static, symmetric=True)
                if is_static:
                    def fn(x, s, _n=norm, _q=quant):
                        return _q.forward_native(_n.forward_native(x), s)
                else:
                    def fn(x, _n=norm, _q=quant):
                        return _q.forward_native(_n.forward_native(x))
            else:
                quant = QuantInt8(static=is_static, symmetric=True)
                if is_static:
                    def fn(x, s, _q=quant):
                        return _q.forward_native(x, s)
                else:
                    def fn(x, _q=quant):
                        return _q.forward_native(x)

            compiled = compile_fn(fn)
            warmup_compile(compiled, inputs)
            results["asm" if use_asm else "fb"] = bench(
                compiled, inputs, args.warmup, args.rep)

        asm_speedup = cuda_ms / results["asm"]
        fb_speedup = cuda_ms / results["fb"]
        print(f"{name:<30} {cuda_ms:>10.4f} {results['asm']:>10.4f}"
              f" {results['fb']:>12.4f}"
              f" {asm_speedup:>11.2f}x{fb_speedup:>11.2f}x")

        csv_rows.append({
            "section": "asm_ablation",
            "op": name,
            "cuda_ms": f"{cuda_ms:.4f}",
            "decomp_ms": "",
            "speedup": "",
            "gpu": gpu_name,
            "shape": f"[{M},{N}]",
            "asm_ms": f"{results['asm']:.4f}",
            "fallback_ms": f"{results['fb']:.4f}",
            "asm_speedup": f"{asm_speedup:.2f}",
            "fb_speedup": f"{fb_speedup:.2f}",
        })

    # Restore
    input_quant_int8._USE_INLINE_ASM = original_use_asm
    return csv_rows


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()

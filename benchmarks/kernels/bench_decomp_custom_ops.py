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
    python benchmarks/kernels/bench_decomp_custom_ops.py --verify
    python benchmarks/kernels/bench_decomp_custom_ops.py --no-compile
"""

import argparse
import csv
import io
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
    compiled = torch.compile(fn, mode="max-autotune-no-cudagraphs")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Benchmark eager decomposition (no Inductor)")
    parser.add_argument("--asm-ablation", action="store_true",
                        help="Compare INT8 inline_asm vs round+clamp fallback")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV file")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    M, N = args.m, args.n
    dtype = torch.bfloat16
    device = "cuda"
    gpu_name = torch.cuda.get_device_name()

    print("vLLM CustomOp Decomposition Benchmark")
    print(f"Shape: [{M}, {N}], dtype={dtype}")
    print(f"Device: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {'compiled' if not args.no_compile else 'eager'}")
    print("=" * 80)

    # Collect all results for CSV
    csv_rows: list[dict[str, str]] = []

    scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    # ========================================================================
    # Section 1: Individual ops
    # ========================================================================
    print("\n--- Individual Ops: forward_cuda vs "
          f"{'compiled ' if not args.no_compile else ''}forward_native ---")
    print(f"{'Op':<40} {'CUDA(ms)':>10} {'Decomp(ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    individual_ops: list[tuple[str, Any, Any, Any]] = []

    # RMSNorm
    cuda_fn, native_fn = make_rmsnorm(N, dtype, device)
    individual_ops.append((
        "rms_norm", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    # FusedAddRMSNorm
    cuda_fn, native_fn = make_fused_add_rmsnorm(N, dtype, device)
    individual_ops.append((
        "fused_add_rms_norm", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),
                 torch.randn(M, N, dtype=dtype, device=device)),
    ))

    # Static FP8
    cuda_fn, native_fn = make_static_fp8_quant(device)
    individual_ops.append((
        "static_fp8_quant", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    # Dynamic FP8 per-tensor
    cuda_fn, native_fn = make_dynamic_fp8_quant_per_tensor(device)
    individual_ops.append((
        "dynamic_fp8_quant_per_tensor", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    # Dynamic FP8 per-token
    cuda_fn, native_fn = make_dynamic_fp8_quant_per_token(device)
    individual_ops.append((
        "dynamic_fp8_quant_per_token", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    # SiluAndMul
    cuda_fn, native_fn = make_silu_and_mul(device)
    individual_ops.append((
        "silu_and_mul", cuda_fn, native_fn,
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device),),
    ))

    # Static INT8
    cuda_fn, native_fn = make_static_int8_quant(device)
    individual_ops.append((
        "static_int8_quant", cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    # Dynamic INT8 per-token
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
            compiled = compile_fn(native_fn)
            warmup_compile(compiled, inputs)
            native_ms = bench(compiled, inputs, args.warmup, args.rep)

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
            "shape": f"[{M},{N}]",
        })

    # ========================================================================
    # Section 2: Fused patterns
    # ========================================================================
    print(f"\n--- Fused Patterns: fused CUDA kernel vs "
          f"{'compiled ' if not args.no_compile else ''}composed "
          f"forward_native ---")
    print(f"{'Pattern':<40} {'CUDA(ms)':>10} {'Decomp(ms)':>10}"
          f" {'Speedup':>10}")
    print("-" * 80)

    fused_ops: list[tuple[str, Any, Any, Any]] = []

    # RMSNorm + static FP8
    cuda_fn, native_fn, label = make_rmsnorm_static_fp8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    # FusedAdd+RMSNorm + static FP8
    cuda_fn, native_fn, label = make_fused_add_rmsnorm_static_fp8(
        N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),
                 torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    # RMSNorm + dynamic per-token FP8
    cuda_fn, native_fn, label = make_rmsnorm_dynamic_per_token_fp8(
        N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device),),
    ))

    # SiluAndMul + static FP8
    cuda_fn, native_fn, label = make_silu_and_mul_static_fp8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device), scale),
    ))

    # RMSNorm + static INT8
    cuda_fn, native_fn, label = make_rmsnorm_static_int8(N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, N, dtype=dtype, device=device), scale),
    ))

    # SiluAndMul + static INT8
    cuda_fn, native_fn, label = make_silu_and_mul_static_int8(
        N, dtype, device)
    fused_ops.append((
        label, cuda_fn, native_fn,
        lambda: (torch.randn(M, 2 * N, dtype=dtype, device=device), scale),
    ))

    # RMSNorm + dynamic INT8
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
            compiled = compile_fn(native_fn)
            warmup_compile(compiled, inputs)
            native_ms = bench(compiled, inputs, args.warmup, args.rep)

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
            "shape": f"[{M},{N}]",
        })

    # ========================================================================
    # Section 3: INT8 inline_asm ablation (optional)
    # ========================================================================
    if args.asm_ablation:
        ablation_rows = bench_int8_inline_asm(args, gpu_name)
        csv_rows.extend(ablation_rows)

    # ========================================================================
    # Write CSV
    # ========================================================================
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        # Check if ablation added extra fields
        for row in csv_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        csv_text = buf.getvalue()

        if args.csv:
            with open(args.csv, "w") as f:
                f.write(csv_text)
            print(f"\nCSV written to {args.csv}")
        else:
            print(f"\n--- CSV ---")
            print(csv_text, end="")


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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that vLLM CustomOp forward_native decompositions match forward_cuda,
both directly and under torch.compile.
"""

import pytest
import torch
import torch.nn.functional as F

M, N = 128, 4096
DTYPE = torch.bfloat16
EPS = 1e-6


@pytest.fixture(autouse=True)
def _cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmsnorm_cuda_native(x, weight):
    from vllm.model_executor.layers.layernorm import RMSNorm
    out_cuda = torch.empty_like(x)
    torch.ops._C.rms_norm(out_cuda, x, weight, EPS)
    out_native = RMSNorm.forward_static(x, EPS, N, DTYPE, weight)
    return out_cuda, out_native


def _fp8_helpers():
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    fp8_dtype = current_platform.fp8_dtype()
    fp8_min, fp8_max = get_fp8_min_max()
    return fp8_dtype, fp8_min, fp8_max


# ---------------------------------------------------------------------------
# Individual op correctness: forward_cuda vs forward_native
# ---------------------------------------------------------------------------

class TestDecompositionCorrectness:

    def test_rmsnorm(self):
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")
        w = torch.randn(N, dtype=DTYPE, device="cuda")
        cuda, native = _rmsnorm_cuda_native(x, w)
        torch.testing.assert_close(cuda, native, atol=1e-2, rtol=1e-2)

    def test_fused_add_rmsnorm(self):
        from vllm.model_executor.layers.layernorm import RMSNorm
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")
        res = torch.randn(M, N, dtype=DTYPE, device="cuda")
        w = torch.randn(N, dtype=DTYPE, device="cuda")

        x_cuda, res_cuda = x.clone(), res.clone()
        torch.ops._C.fused_add_rms_norm(x_cuda, res_cuda, w, EPS)
        normed, res_native = RMSNorm.forward_static(x, EPS, N, DTYPE, w, res)

        torch.testing.assert_close(x_cuda, normed, atol=0.02, rtol=0.02)
        torch.testing.assert_close(res_cuda, res_native, atol=0.02, rtol=0.02)

    def test_silu_and_mul(self):
        x = torch.randn(M, 2 * N, dtype=DTYPE, device="cuda")
        d = N
        out_cuda = torch.empty(M, d, dtype=DTYPE, device="cuda")
        torch.ops._C.silu_and_mul(out_cuda, x)
        out_native = F.silu(x[..., :d]) * x[..., d:]
        torch.testing.assert_close(out_cuda, out_native, atol=1e-2, rtol=1e-2)

    def test_static_fp8(self):
        fp8, fp8_min, fp8_max = _fp8_helpers()
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

        out_cuda = torch.empty_like(x, dtype=fp8)
        torch.ops._C.static_scaled_fp8_quant(out_cuda, x, scale)
        out_native = (x.float() * scale.float().reciprocal()).clamp(
            fp8_min, fp8_max
        ).to(fp8)
        torch.testing.assert_close(
            out_cuda.float(), out_native.float(), atol=1.0, rtol=0.05
        )

    def test_dynamic_fp8(self):
        fp8, fp8_min, fp8_max = _fp8_helpers()
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")

        out_cuda = torch.empty_like(x, dtype=fp8)
        scale_cuda = torch.empty(1, device="cuda", dtype=torch.float32)
        torch.ops._C.dynamic_scaled_fp8_quant(out_cuda, x, scale_cuda)

        x_max = x.abs().max().unsqueeze(-1).float()
        scale_native = (x_max / fp8_max).clamp(min=1.0 / (fp8_max * 512.0))
        out_native = (x.float() * scale_native.reciprocal()).clamp(
            fp8_min, fp8_max
        ).to(fp8)

        torch.testing.assert_close(scale_cuda, scale_native, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            out_cuda.float(), out_native.float(), atol=1.0, rtol=0.05
        )

    def test_static_int8(self, default_vllm_config):
        import vllm._custom_ops as ops
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")
        scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")

        out_cuda, _, _ = ops.scaled_int8_quant(x, scale)

        @torch.compile
        def compiled(x, s):
            return ops.scaled_int8_quant(x, s)
        out_compiled, _, _ = compiled(x, scale)

        torch.testing.assert_close(out_cuda.float(), out_compiled.float(),
                                   atol=0, rtol=0)

    def test_dynamic_int8(self, default_vllm_config):
        import vllm._custom_ops as ops
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")

        out_cuda, scale_cuda, _ = ops.scaled_int8_quant(x)

        @torch.compile
        def compiled(x):
            return ops.scaled_int8_quant(x)
        out_compiled, scale_compiled, _ = compiled(x)

        torch.testing.assert_close(scale_cuda, scale_compiled, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out_cuda.float(), out_compiled.float(),
                                   atol=1, rtol=0.01)


# ---------------------------------------------------------------------------
# Fused patterns: compiled composed decompositions vs CUDA
# ---------------------------------------------------------------------------

class TestFusedPatterns:

    def test_rmsnorm_fp8(self):
        fp8, fp8_min, fp8_max = _fp8_helpers()
        from vllm.model_executor.layers.layernorm import RMSNorm

        x = torch.randn(M, N, dtype=DTYPE, device="cuda")
        w = torch.randn(N, dtype=DTYPE, device="cuda")
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

        expected = torch.empty_like(x, dtype=fp8)
        torch.ops._C.rms_norm_static_fp8_quant(expected, x, w, scale, EPS)

        @torch.compile
        def fused(x, w, s):
            normed = RMSNorm.forward_static(x, EPS, N, DTYPE, w)
            return (normed.float() * s.float().reciprocal()).clamp(
                fp8_min, fp8_max
            ).to(fp8)

        torch.testing.assert_close(
            fused(x, w, scale).float(), expected.float(), atol=2.0, rtol=0.1
        )

    def test_silu_fp8(self):
        fp8, fp8_min, fp8_max = _fp8_helpers()
        x = torch.randn(M, 2 * N, dtype=DTYPE, device="cuda")
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

        d = N
        silu_out = torch.empty(M, d, dtype=DTYPE, device="cuda")
        torch.ops._C.silu_and_mul(silu_out, x)
        expected = torch.empty_like(silu_out, dtype=fp8)
        torch.ops._C.static_scaled_fp8_quant(expected, silu_out, scale)

        @torch.compile
        def fused(x, s):
            d = x.shape[-1] // 2
            act = F.silu(x[..., :d]) * x[..., d:]
            return (act.float() * s.float().reciprocal()).clamp(
                fp8_min, fp8_max
            ).to(fp8)

        torch.testing.assert_close(
            fused(x, scale).float(), expected.float(), atol=2.0, rtol=0.1
        )


# ---------------------------------------------------------------------------
# Config logic
# ---------------------------------------------------------------------------

class TestConfigLogic:

    def test_norm_fusion_disabled(self):
        from unittest.mock import MagicMock
        from vllm.config.vllm import enable_norm_fusion
        cfg = MagicMock()
        assert enable_norm_fusion(cfg) is False

    def test_act_fusion_disabled_non_nvfp4(self):
        from unittest.mock import MagicMock
        from vllm.config.vllm import enable_act_fusion
        cfg = MagicMock()
        cfg.model_config.is_nvfp4_quantized.return_value = False
        assert enable_act_fusion(cfg) is False

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that vLLM CustomOp decompositions produce correct results
when CustomOps are disabled (compilation_config.custom_ops = ["none"]).

Verifies:
1. RMSNorm.forward_native matches forward_cuda
2. QuantFP8.forward_native matches forward_cuda
3. SiluAndMul.forward_native matches forward_cuda
4. Compiled decompositions produce correct results
5. Fused patterns (RMSNorm + FP8Quant) produce correct results
"""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture
def dtype():
    return torch.bfloat16


@pytest.fixture
def shape():
    return (128, 4096)


class TestRMSNormDecomposition:
    """Test RMSNorm forward_native matches forward_cuda."""

    def test_rmsnorm_correctness(self, device, dtype, shape):
        M, N = shape
        eps = 1e-6

        x = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)

        # CUDA kernel
        output_cuda = torch.empty_like(x)
        torch.ops._C.rms_norm(output_cuda, x, weight, eps)

        # Native decomposition
        from vllm.model_executor.layers.layernorm import RMSNorm
        output_native = RMSNorm.forward_static(x, eps, N, dtype, weight)

        torch.testing.assert_close(
            output_cuda, output_native, atol=1e-2, rtol=1e-2
        )

    def test_fused_add_rmsnorm_correctness(self, device, dtype, shape):
        M, N = shape
        eps = 1e-6

        x = torch.randn(M, N, dtype=dtype, device=device)
        residual = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)

        # CUDA kernel (in-place)
        x_cuda = x.clone()
        res_cuda = residual.clone()
        torch.ops._C.fused_add_rms_norm(x_cuda, res_cuda, weight, eps)

        # Native decomposition
        from vllm.model_executor.layers.layernorm import RMSNorm
        normed_native, res_native = RMSNorm.forward_static(
            x, eps, N, dtype, weight, residual
        )

        # bf16 rounding causes small diffs (~0.008)
        torch.testing.assert_close(
            x_cuda, normed_native, atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(
            res_cuda, res_native, atol=0.02, rtol=0.02
        )

    def test_rmsnorm_compiled(self, device, dtype, shape):
        M, N = shape
        eps = 1e-6

        x = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)

        from vllm.model_executor.layers.layernorm import RMSNorm

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def compiled_rmsnorm(x, weight):
            return RMSNorm.forward_static(x, eps, N, dtype, weight)

        output = compiled_rmsnorm(x, weight)

        # Compare with CUDA
        expected = torch.empty_like(x)
        torch.ops._C.rms_norm(expected, x, weight, eps)

        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


class TestQuantFP8Decomposition:
    """Test QuantFP8 forward_native matches forward_cuda."""

    def test_static_fp8_quant_correctness(self, device, dtype, shape):
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        fp8_dtype = current_platform.fp8_dtype()
        fp8_min, fp8_max = get_fp8_min_max()
        fp8_min_scaling = 1.0 / (fp8_max * 512.0)

        M, N = shape
        x = torch.randn(M, N, dtype=dtype, device=device)
        scale = torch.tensor([0.1], dtype=torch.float32, device=device)

        # CUDA kernel
        output_cuda = torch.empty_like(x, dtype=fp8_dtype)
        torch.ops._C.static_scaled_fp8_quant(output_cuda, x, scale)

        # Native decomposition
        out_native = x.to(torch.float32) * scale.to(torch.float32).reciprocal()
        output_native = out_native.clamp(fp8_min, fp8_max).to(fp8_dtype)

        torch.testing.assert_close(
            output_cuda.float(), output_native.float(), atol=1.0, rtol=0.05
        )

    def test_dynamic_fp8_quant_correctness(self, device, dtype, shape):
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        fp8_dtype = current_platform.fp8_dtype()
        fp8_min, fp8_max = get_fp8_min_max()
        fp8_min_scaling = 1.0 / (fp8_max * 512.0)

        M, N = shape
        x = torch.randn(M, N, dtype=dtype, device=device)

        # CUDA kernel
        output_cuda = torch.empty_like(x, dtype=fp8_dtype)
        scale_cuda = torch.empty(1, device=device, dtype=torch.float32)
        torch.ops._C.dynamic_scaled_fp8_quant(output_cuda, x, scale_cuda)

        # Native decomposition
        x_max = x.abs().max().unsqueeze(-1).to(torch.float32)
        scale_native = (x_max / fp8_max).clamp(min=fp8_min_scaling)
        out_native = x.to(torch.float32) * scale_native.reciprocal()
        output_native = out_native.clamp(fp8_min, fp8_max).to(fp8_dtype)

        torch.testing.assert_close(
            scale_cuda, scale_native, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            output_cuda.float(), output_native.float(), atol=1.0, rtol=0.05
        )

    def test_static_fp8_compiled(self, device, dtype, shape):
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        fp8_dtype = current_platform.fp8_dtype()
        fp8_min, fp8_max = get_fp8_min_max()

        M, N = shape
        x = torch.randn(M, N, dtype=dtype, device=device)
        scale = torch.tensor([0.1], dtype=torch.float32, device=device)

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def compiled_static_fp8(x, scale):
            out = x.to(torch.float32) * scale.to(torch.float32).reciprocal()
            return out.clamp(fp8_min, fp8_max).to(fp8_dtype)

        output = compiled_static_fp8(x, scale)

        # Compare with CUDA
        expected = torch.empty_like(x, dtype=fp8_dtype)
        torch.ops._C.static_scaled_fp8_quant(expected, x, scale)

        torch.testing.assert_close(
            output.float(), expected.float(), atol=1.0, rtol=0.05
        )

    def test_direct_vllm_fp8_decomposition(self, device, dtype, shape):
        """Test that vllm_ops.scaled_fp8_quant uses decomposition when compiled."""
        import vllm._custom_ops as vllm_ops

        M, N = shape
        x = torch.randn(M, N, dtype=dtype, device=device)
        scale = torch.tensor([0.1], dtype=torch.float32, device=device)

        # Reference: vLLM FP8 without compilation (uses CUDA)
        output_cuda, _ = vllm_ops.scaled_fp8_quant(x, scale=scale)

        # Compiled: vLLM FP8 with compilation (should use decomposition)
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def compiled_vllm_fp8_quant(x, scale):
            return vllm_ops.scaled_fp8_quant(x, scale=scale)

        # Warmup compilation
        _ = compiled_vllm_fp8_quant(x, scale)
        output_decomp, _ = compiled_vllm_fp8_quant(x, scale)

        # Should be close (decomposition should match CUDA)
        torch.testing.assert_close(
            output_cuda.float(), output_decomp.float(), atol=1.0, rtol=0.05
        )

    def test_rmsnorm_plus_direct_static_fp8_fusion(self, device, dtype, shape):
        """Test RMSNorm + direct vllm_ops.scaled_fp8_quant fusion using our decompositions."""
        import vllm._custom_ops as vllm_ops
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.layernorm import RMSNorm

        fp8_dtype = current_platform.fp8_dtype()

        M, N = shape
        eps = 1e-6
        x = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        scale = torch.tensor([0.1], dtype=torch.float32, device=device)

        # Reference: sequential operations without compilation
        normed_ref = RMSNorm.forward_static(x, eps, N, dtype, weight)
        quant_ref, _ = vllm_ops.scaled_fp8_quant(normed_ref, scale=scale)

        # Compiled fusion: RMSNorm + vLLM FP8 (should use decompositions & fuse)
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def fused_rmsnorm_vllm_fp8(x, weight, scale):
            # RMSNorm decomposition
            normed = RMSNorm.forward_static(x, eps, N, dtype, weight)
            # vLLM FP8 quantization - should use decomposition when compiled
            quant_out, _ = vllm_ops.scaled_fp8_quant(normed, scale=scale)
            return quant_out

        # Warmup compilation
        _ = fused_rmsnorm_vllm_fp8(x, weight, scale)
        output_fused = fused_rmsnorm_vllm_fp8(x, weight, scale)

        # Should match reference (both decompositions should fuse into fewer kernels)
        torch.testing.assert_close(
            quant_ref.float(), output_fused.float(), atol=2.0, rtol=0.1
        )

    def test_direct_vllm_int8_decomposition(self, device, dtype, shape, default_vllm_config):
        """Test that vllm_ops.scaled_int8_quant uses PTX decomposition when compiled."""
        import vllm._custom_ops as vllm_ops

        M, N = shape
        x = torch.randn(M, N, dtype=dtype, device=device)
        scale = torch.tensor([0.05], dtype=torch.float32, device=device)

        # Reference: vLLM INT8 without compilation (uses CUDA)
        output_cuda, _, _ = vllm_ops.scaled_int8_quant(x, scale=scale)

        # Compiled: vLLM INT8 with compilation (should use PTX Triton kernel)
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def compiled_vllm_int8_quant(x, scale):
            return vllm_ops.scaled_int8_quant(x, scale=scale)

        # Warmup compilation
        _ = compiled_vllm_int8_quant(x, scale)
        output_decomp, _, _ = compiled_vllm_int8_quant(x, scale)

        # Should match exactly (PTX instruction should be bit-identical to CUDA)
        torch.testing.assert_close(
            output_cuda.float(), output_decomp.float(), atol=0, rtol=0
        )

    def test_dynamic_vllm_int8_decomposition(self, device, dtype, shape, default_vllm_config):
        """Test dynamic INT8 quantization uses decomposition when compiled."""
        import vllm._custom_ops as vllm_ops

        M, N = shape
        x = torch.randn(M, N, dtype=dtype, device=device)

        # Reference: dynamic INT8 without compilation
        output_cuda, scale_cuda, _ = vllm_ops.scaled_int8_quant(x)

        # Compiled: dynamic INT8 with compilation (should use decomposition)
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def compiled_dynamic_int8_quant(x):
            return vllm_ops.scaled_int8_quant(x)

        # Warmup compilation
        _ = compiled_dynamic_int8_quant(x)
        output_decomp, scale_decomp, _ = compiled_dynamic_int8_quant(x)

        # Should match the CUDA version closely (dynamic scaling may have slight differences)
        torch.testing.assert_close(
            output_cuda.float(), output_decomp.float(), atol=1, rtol=0.01
        )
        torch.testing.assert_close(
            scale_cuda, scale_decomp, atol=1e-4, rtol=1e-4
        )


class TestSiluAndMulDecomposition:
    """Test SiluAndMul forward_native matches forward_cuda."""

    def test_silu_and_mul_correctness(self, device, dtype, shape):
        import torch.nn.functional as F

        M, N = shape
        x = torch.randn(M, 2 * N, dtype=dtype, device=device)

        # CUDA kernel
        d = x.shape[-1] // 2
        output_cuda = torch.empty(x.shape[:-1] + (d,), dtype=dtype, device=device)
        torch.ops._C.silu_and_mul(output_cuda, x)

        # Native decomposition
        output_native = F.silu(x[..., :d]) * x[..., d:]

        torch.testing.assert_close(
            output_cuda, output_native, atol=1e-2, rtol=1e-2
        )


class TestFusedPatterns:
    """Test that Inductor-fused patterns produce correct results."""

    def test_rmsnorm_plus_fp8quant_fused(self, device, dtype, shape):
        """RMSNorm + FP8 quant should produce correct results when compiled."""
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        fp8_dtype = current_platform.fp8_dtype()
        fp8_min, fp8_max = get_fp8_min_max()

        M, N = shape
        eps = 1e-6
        x = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        scale = torch.tensor([0.1], dtype=torch.float32, device=device)

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def fused_rmsnorm_fp8(x, weight, scale):
            normed = RMSNorm.forward_static(x, eps, N, dtype, weight)
            out = normed.to(torch.float32) * scale.to(torch.float32).reciprocal()
            return out.clamp(fp8_min, fp8_max).to(fp8_dtype)

        # Warmup compilation
        _ = fused_rmsnorm_fp8(x, weight, scale)
        output = fused_rmsnorm_fp8(x, weight, scale)

        # Compare with fused CUDA kernel
        expected = torch.empty_like(x, dtype=fp8_dtype)
        torch.ops._C.rms_norm_static_fp8_quant(
            expected, x, weight, scale, eps
        )

        # FP8 has limited precision; compare in float with appropriate tolerance
        torch.testing.assert_close(
            output.float(), expected.float(), atol=2.0, rtol=0.1
        )

    def test_silu_and_mul_plus_fp8quant_fused(self, device, dtype, shape):
        """SiluAndMul + FP8 quant should produce correct results when compiled."""
        import torch.nn.functional as F
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        fp8_dtype = current_platform.fp8_dtype()
        fp8_min, fp8_max = get_fp8_min_max()

        M, N = shape
        x = torch.randn(M, 2 * N, dtype=dtype, device=device)
        scale = torch.tensor([0.1], dtype=torch.float32, device=device)

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def fused_silu_fp8(x, scale):
            d = x.shape[-1] // 2
            activated = F.silu(x[..., :d]) * x[..., d:]
            out = activated.to(torch.float32) * scale.to(torch.float32).reciprocal()
            return out.clamp(fp8_min, fp8_max).to(fp8_dtype)

        # Warmup compilation
        _ = fused_silu_fp8(x, scale)
        output = fused_silu_fp8(x, scale)

        # Compare with sequential CUDA
        d = x.shape[-1] // 2
        silu_out = torch.empty(x.shape[:-1] + (d,), dtype=dtype, device=device)
        torch.ops._C.silu_and_mul(silu_out, x)
        expected = torch.empty_like(silu_out, dtype=fp8_dtype)
        torch.ops._C.static_scaled_fp8_quant(expected, silu_out, scale)

        # FP8 has limited precision; compare in float with appropriate tolerance
        torch.testing.assert_close(
            output.float(), expected.float(), atol=2.0, rtol=0.1
        )


class TestConfigurationLogic:
    """Test that vLLM configuration auto-disables fusion when CustomOps disabled."""

    def test_enable_norm_fusion_disabled_when_no_custom_ops(self):
        """enable_norm_fusion should return False when custom_ops=["none"]."""
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.compilation_config.is_custom_op_enabled.return_value = False

        from vllm.config.vllm import enable_norm_fusion
        result = enable_norm_fusion(cfg)
        assert result is False, "Norm fusion should be disabled when CustomOps disabled"

    def test_enable_act_fusion_disabled_when_no_custom_ops(self):
        """enable_act_fusion should return False when custom_ops=["none"]."""
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.compilation_config.is_custom_op_enabled.return_value = False
        cfg.model_config.is_nvfp4_quantized.return_value = False

        from vllm.config.vllm import enable_act_fusion
        result = enable_act_fusion(cfg)
        assert result is False, "Act fusion should be disabled when CustomOps disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

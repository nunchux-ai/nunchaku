"""Benchmark tests for fast LoRA switching performance.

This module tests the performance difference between traditional LoRA switching
(reset_lora + update_lora_params) and fast LoRA switching (preload_loras + switch_lora).

Run with: pytest tests/flux/test_fast_lora_switching_benchmark.py -v -s
"""

import gc
import logging
import time
from dataclasses import dataclass

import pytest
import torch

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

_LOGGER = logging.getLogger(__name__)

# Expected speedup ratios for different GPUs
_EXPECTED_SPEEDUPS = {
    "NVIDIA GeForce RTX 3090": 30.0,
    "NVIDIA GeForce RTX 4090": 40.0,
    "NVIDIA GeForce RTX 5090": 40.0,
}


@dataclass
class TimingResult:
    """Timing result for a single LoRA switch + forward pass."""

    lora_name: str
    switch_time_ms: float
    forward_time_ms: float

    @property
    def total_time_ms(self) -> float:
        return self.switch_time_ms + self.forward_time_ms


def create_dummy_inputs(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> dict:
    """Create dummy inputs for transformer forward pass."""
    return {
        "hidden_states": torch.randn(1, 256, 64, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(1, 512, 4096, device=device, dtype=dtype),
        "pooled_projections": torch.randn(1, 768, device=device, dtype=dtype),
        "timestep": torch.tensor([500.0], device=device, dtype=dtype),
        "img_ids": torch.zeros(256, 3, device=device, dtype=dtype),
        "txt_ids": torch.zeros(512, 3, device=device, dtype=dtype),
        "guidance": torch.tensor([3.5], device=device, dtype=dtype),
        "return_dict": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFastLoRASwitchingBenchmark:
    """Benchmark tests for fast LoRA switching."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clean up GPU memory before and after each test."""
        gc.collect()
        torch.cuda.empty_cache()
        yield
        gc.collect()
        torch.cuda.empty_cache()

    def test_switch_speedup(self):
        """Test that fast LoRA switching is significantly faster than traditional method."""
        precision = get_precision()
        device_name = torch.cuda.get_device_name(0)
        expected_speedup = _EXPECTED_SPEEDUPS.get(device_name, 20.0)

        _LOGGER.info(f"GPU: {device_name}, Precision: {precision}")

        # Load model
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
        )

        # Get LoRA paths
        from huggingface_hub import hf_hub_download

        lora_anime = hf_hub_download("XLabs-AI/flux-lora-collection", "anime_lora.safetensors")
        lora_realism = hf_hub_download("XLabs-AI/flux-lora-collection", "realism_lora.safetensors")

        loras = {"anime": lora_anime, "realism": lora_realism}
        lora_names = list(loras.keys())
        num_iters = 6

        # Warm up
        inputs = create_dummy_inputs()
        for _ in range(3):
            with torch.no_grad():
                _ = transformer(**inputs)
        torch.cuda.synchronize()

        # Benchmark traditional method
        transformer.reset_lora()
        traditional_switch_times = []

        for i in range(num_iters):
            lora_name = lora_names[i % 2]
            lora_path = loras[lora_name]

            transformer.reset_lora()
            torch.cuda.synchronize()

            start = time.perf_counter()
            transformer.update_lora_params(lora_path)
            torch.cuda.synchronize()
            traditional_switch_times.append((time.perf_counter() - start) * 1000)

        traditional_avg = sum(traditional_switch_times) / len(traditional_switch_times)
        _LOGGER.info(f"Traditional switch avg: {traditional_avg:.2f}ms")

        # Reset for fast method
        transformer.reset_lora()
        gc.collect()
        torch.cuda.empty_cache()

        # Preload LoRAs
        preload_start = time.perf_counter()
        transformer.preload_loras(loras)
        torch.cuda.synchronize()
        preload_time = (time.perf_counter() - preload_start) * 1000
        _LOGGER.info(f"Preload time: {preload_time:.2f}ms")

        # Benchmark fast method
        # Warm up
        for name in lora_names:
            transformer.switch_lora(name)
        torch.cuda.synchronize()

        fast_switch_times = []
        for i in range(num_iters):
            lora_name = lora_names[i % 2]

            start = time.perf_counter()
            transformer.switch_lora(lora_name)
            torch.cuda.synchronize()
            fast_switch_times.append((time.perf_counter() - start) * 1000)

        fast_avg = sum(fast_switch_times) / len(fast_switch_times)
        _LOGGER.info(f"Fast switch avg: {fast_avg:.2f}ms")

        speedup = traditional_avg / fast_avg
        _LOGGER.info(f"Speedup: {speedup:.1f}x")

        # Cleanup
        transformer.clear_preloaded_loras()
        del transformer

        assert speedup >= expected_speedup * 0.5, (
            f"Fast switching should be at least {expected_speedup * 0.5:.1f}x faster, " f"but got {speedup:.1f}x"
        )

    def test_forward_pass_consistency(self):
        """Test that forward pass time is consistent between switching methods."""
        precision = get_precision()

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
        )

        from huggingface_hub import hf_hub_download

        lora_path = hf_hub_download("XLabs-AI/flux-lora-collection", "anime_lora.safetensors")
        inputs = create_dummy_inputs()

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = transformer(**inputs)
        torch.cuda.synchronize()

        # Test traditional method forward time
        transformer.update_lora_params(lora_path)
        torch.cuda.synchronize()

        traditional_times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = transformer(**inputs)
            torch.cuda.synchronize()
            traditional_times.append((time.perf_counter() - start) * 1000)

        traditional_avg = sum(traditional_times) / len(traditional_times)

        # Reset and use fast method
        transformer.reset_lora()
        transformer.preload_loras({"anime": lora_path})
        transformer.switch_lora("anime")
        torch.cuda.synchronize()

        fast_times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = transformer(**inputs)
            torch.cuda.synchronize()
            fast_times.append((time.perf_counter() - start) * 1000)

        fast_avg = sum(fast_times) / len(fast_times)

        _LOGGER.info(f"Traditional forward avg: {traditional_avg:.2f}ms")
        _LOGGER.info(f"Fast forward avg: {fast_avg:.2f}ms")

        # Forward times should be within 20% of each other
        ratio = max(traditional_avg, fast_avg) / min(traditional_avg, fast_avg)
        assert ratio < 1.2, f"Forward times should be similar, but ratio is {ratio:.2f}"

        # Cleanup
        transformer.clear_preloaded_loras()
        del transformer

    def test_inference_with_switching(self):
        """Test complete inference workflow with LoRA switching."""
        precision = get_precision()

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
        )

        from huggingface_hub import hf_hub_download

        lora_anime = hf_hub_download("XLabs-AI/flux-lora-collection", "anime_lora.safetensors")
        lora_realism = hf_hub_download("XLabs-AI/flux-lora-collection", "realism_lora.safetensors")

        loras = {"anime": lora_anime, "realism": lora_realism, "base": None}
        inputs = create_dummy_inputs()
        num_iters = 9  # 3 cycles through 3 LoRAs

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = transformer(**inputs)
        torch.cuda.synchronize()

        # Traditional method
        transformer.reset_lora()
        traditional_results: list[TimingResult] = []

        lora_names = list(loras.keys())
        for i in range(num_iters):
            lora_name = lora_names[i % 3]
            lora_path = loras[lora_name]

            transformer.reset_lora()
            torch.cuda.synchronize()

            switch_start = time.perf_counter()
            if lora_path is not None:
                transformer.update_lora_params(lora_path)
            torch.cuda.synchronize()
            switch_time = (time.perf_counter() - switch_start) * 1000

            forward_start = time.perf_counter()
            with torch.no_grad():
                _ = transformer(**inputs)
            torch.cuda.synchronize()
            forward_time = (time.perf_counter() - forward_start) * 1000

            traditional_results.append(TimingResult(lora_name, switch_time, forward_time))

        # Fast method
        transformer.reset_lora()
        gc.collect()
        torch.cuda.empty_cache()

        preload_start = time.perf_counter()
        transformer.preload_loras(loras)
        torch.cuda.synchronize()
        preload_time = (time.perf_counter() - preload_start) * 1000

        # Warm up fast switches
        for name in lora_names:
            transformer.switch_lora(name)
        torch.cuda.synchronize()

        fast_results: list[TimingResult] = []

        for i in range(num_iters):
            lora_name = lora_names[i % 3]

            switch_start = time.perf_counter()
            transformer.switch_lora(lora_name)
            torch.cuda.synchronize()
            switch_time = (time.perf_counter() - switch_start) * 1000

            forward_start = time.perf_counter()
            with torch.no_grad():
                _ = transformer(**inputs)
            torch.cuda.synchronize()
            forward_time = (time.perf_counter() - forward_start) * 1000

            fast_results.append(TimingResult(lora_name, switch_time, forward_time))

        # Print comparison table
        _LOGGER.info("\nPer-iteration comparison:")
        _LOGGER.info(f"{'#':<3} {'LoRA':<10} {'Trad Switch':>12} {'Fast Switch':>12} {'Saved':>10}")
        _LOGGER.info("-" * 55)
        for i, (t, f) in enumerate(zip(traditional_results, fast_results)):
            saved = t.switch_time_ms - f.switch_time_ms
            _LOGGER.info(
                f"{i+1:<3} {t.lora_name:<10} {t.switch_time_ms:>10.1f}ms {f.switch_time_ms:>10.1f}ms {saved:>8.1f}ms"
            )

        # Calculate totals
        trad_total_switch = sum(r.switch_time_ms for r in traditional_results)
        trad_total_forward = sum(r.forward_time_ms for r in traditional_results)
        fast_total_switch = sum(r.switch_time_ms for r in fast_results)
        fast_total_forward = sum(r.forward_time_ms for r in fast_results)

        _LOGGER.info(f"\nSummary ({num_iters} iterations):")
        _LOGGER.info(f"  Traditional - switch: {trad_total_switch:.1f}ms, forward: {trad_total_forward:.1f}ms")
        _LOGGER.info(f"  Fast        - switch: {fast_total_switch:.1f}ms, forward: {fast_total_forward:.1f}ms")
        _LOGGER.info(f"  Preload cost: {preload_time:.1f}ms")
        _LOGGER.info(f"  Time saved (excl preload): {trad_total_switch - fast_total_switch:.1f}ms")
        _LOGGER.info(f"  Time saved (incl preload): {trad_total_switch - fast_total_switch - preload_time:.1f}ms")

        # Assertions
        assert fast_total_switch < trad_total_switch, "Fast method should have lower total switch time"
        assert (trad_total_switch - fast_total_switch) > preload_time, (
            f"Switch time savings ({trad_total_switch - fast_total_switch:.1f}ms) should exceed "
            f"preload cost ({preload_time:.1f}ms) for {num_iters} iterations"
        )

        # Cleanup
        transformer.clear_preloaded_loras()
        del transformer

    def test_output_consistency(self):
        """Test that fast switching produces identical output as traditional switching."""
        precision = get_precision()

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
        )

        from huggingface_hub import hf_hub_download

        lora_path = hf_hub_download("XLabs-AI/flux-lora-collection", "anime_lora.safetensors")

        inputs = create_dummy_inputs()

        # Traditional method output
        transformer.update_lora_params(lora_path)
        torch.cuda.synchronize()

        with torch.no_grad():
            output_traditional = transformer(**inputs)
        if isinstance(output_traditional, tuple):
            output_traditional = output_traditional[0]
        output_traditional = output_traditional.clone()

        # Reset and use fast method
        transformer.reset_lora()
        transformer.preload_loras({"anime": lora_path})
        transformer.switch_lora("anime")
        torch.cuda.synchronize()

        with torch.no_grad():
            output_fast = transformer(**inputs)
        if isinstance(output_fast, tuple):
            output_fast = output_fast[0]

        # Compare outputs
        max_diff = (output_traditional - output_fast).abs().max().item()
        mean_diff = (output_traditional - output_fast).abs().mean().item()

        _LOGGER.info(f"Output comparison - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")

        # Outputs should be nearly identical (allow small floating point differences)
        assert max_diff < 1e-4, f"Outputs differ too much: max_diff={max_diff}"
        assert mean_diff < 1e-5, f"Outputs differ too much: mean_diff={mean_diff}"

        # Cleanup
        transformer.clear_preloaded_loras()
        del transformer

    def test_output_differs_between_loras(self):
        """Test that different LoRAs produce different outputs (sanity check)."""
        precision = get_precision()

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
        )

        from huggingface_hub import hf_hub_download

        lora_anime = hf_hub_download("XLabs-AI/flux-lora-collection", "anime_lora.safetensors")
        lora_realism = hf_hub_download("XLabs-AI/flux-lora-collection", "realism_lora.safetensors")

        # Preload both LoRAs
        transformer.preload_loras(
            {
                "anime": lora_anime,
                "realism": lora_realism,
                "base": None,
            }
        )

        inputs = create_dummy_inputs()
        outputs = {}

        for name in ["anime", "realism", "base"]:
            transformer.switch_lora(name)
            torch.cuda.synchronize()

            with torch.no_grad():
                output = transformer(**inputs)
            if isinstance(output, tuple):
                output = output[0]
            outputs[name] = output.clone()

        # Compare different LoRAs - they should produce different outputs
        diff_anime_realism = (outputs["anime"] - outputs["realism"]).abs().mean().item()
        diff_anime_base = (outputs["anime"] - outputs["base"]).abs().mean().item()
        diff_realism_base = (outputs["realism"] - outputs["base"]).abs().mean().item()

        _LOGGER.info("Output differences:")
        _LOGGER.info(f"  anime vs realism: {diff_anime_realism:.6f}")
        _LOGGER.info(f"  anime vs base:    {diff_anime_base:.6f}")
        _LOGGER.info(f"  realism vs base:  {diff_realism_base:.6f}")

        # Different LoRAs should produce meaningfully different outputs
        assert diff_anime_realism > 0.01, "anime and realism LoRAs should produce different outputs"
        assert diff_anime_base > 0.01, "anime LoRA should differ from base"
        assert diff_realism_base > 0.01, "realism LoRA should differ from base"

        # Cleanup
        transformer.clear_preloaded_loras()
        del transformer

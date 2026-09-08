"""Tests for fast LoRA switching functionality."""

import gc
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


# Mock the nunchaku._C module before importing any nunchaku modules
# This allows unit tests to run without the C extension
@pytest.fixture(autouse=True)
def mock_c_extension():
    """Mock the C extension module to allow testing without GPU."""
    mock_c = MagicMock()
    mock_c.QuantizedFluxModel = MagicMock
    mock_c.utils = MagicMock()

    # Store original modules to restore later
    original_modules = {}
    modules_to_mock = ["nunchaku._C", "nunchaku._C.ops", "nunchaku._C.utils"]

    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
        sys.modules[mod_name] = mock_c

    yield

    # Restore original modules
    for mod_name in modules_to_mock:
        if mod_name in original_modules:
            sys.modules[mod_name] = original_modules[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


class TestFastLoRASwitchingUnit:
    """Unit tests for fast LoRA switching (no GPU required)."""

    def test_preload_loras_initializes_variants(self, mock_c_extension):
        """Test that preload_loras populates the variant dictionaries."""
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        # Create a minimal mock transformer
        with patch.object(NunchakuFluxTransformer2dModel, "__init__", lambda self: None):
            transformer = NunchakuFluxTransformer2dModel()
            transformer._lora_variants = {}
            transformer._lora_variant_unquantized = {}
            transformer._lora_variant_vectors = {}
            transformer._active_lora_variant = None
            transformer._quantized_part_sd = {"lora_down": torch.zeros(1), "lora_up": torch.zeros(1)}
            transformer._unquantized_part_sd = {}

            # Mock the parameters() to return a tensor on CPU for device detection
            transformer.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

            # Test preloading None (base model)
            transformer.preload_loras({"none": None})

            assert "none" in transformer._lora_variants
            assert "none" in transformer._lora_variant_unquantized
            assert "none" in transformer._lora_variant_vectors

    def test_list_preloaded_loras(self, mock_c_extension):
        """Test that list_preloaded_loras returns correct names."""
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        with patch.object(NunchakuFluxTransformer2dModel, "__init__", lambda self: None):
            transformer = NunchakuFluxTransformer2dModel()
            transformer._lora_variants = {"a": {}, "b": {}, "c": {}}

            result = transformer.list_preloaded_loras()
            assert set(result) == {"a", "b", "c"}

    def test_get_active_lora_returns_none_initially(self, mock_c_extension):
        """Test that get_active_lora returns None when no variant is active."""
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        with patch.object(NunchakuFluxTransformer2dModel, "__init__", lambda self: None):
            transformer = NunchakuFluxTransformer2dModel()
            transformer._active_lora_variant = None

            assert transformer.get_active_lora() is None

    def test_switch_lora_raises_on_unknown_variant(self, mock_c_extension):
        """Test that switch_lora raises KeyError for unknown variants."""
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        with patch.object(NunchakuFluxTransformer2dModel, "__init__", lambda self: None):
            transformer = NunchakuFluxTransformer2dModel()
            transformer._lora_variants = {"known": {}}

            with pytest.raises(KeyError, match="unknown"):
                transformer.switch_lora("unknown")

    def test_unload_lora_variant_raises_on_active(self, mock_c_extension):
        """Test that unload_lora_variant raises when trying to unload active variant."""
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        with patch.object(NunchakuFluxTransformer2dModel, "__init__", lambda self: None):
            transformer = NunchakuFluxTransformer2dModel()
            transformer._lora_variants = {"active": {}}
            transformer._lora_variant_unquantized = {"active": {}}
            transformer._lora_variant_vectors = {"active": {}}
            transformer._active_lora_variant = "active"

            with pytest.raises(ValueError, match="Cannot unload active"):
                transformer.unload_lora_variant("active")

    def test_unload_lora_variant_removes_variant(self, mock_c_extension):
        """Test that unload_lora_variant removes the variant."""
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        with patch.object(NunchakuFluxTransformer2dModel, "__init__", lambda self: None):
            transformer = NunchakuFluxTransformer2dModel()
            transformer._lora_variants = {"a": {}, "b": {}}
            transformer._lora_variant_unquantized = {"a": {}, "b": {}}
            transformer._lora_variant_vectors = {"a": {}, "b": {}}
            transformer._active_lora_variant = "a"

            transformer.unload_lora_variant("b")

            assert "b" not in transformer._lora_variants
            assert "a" in transformer._lora_variants


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFastLoRASwitchingIntegration:
    """Integration tests for fast LoRA switching (requires GPU)."""

    def test_preload_and_switch_loras(self):
        """Test full preload and switch workflow."""
        from nunchaku import NunchakuFluxTransformer2dModel
        from nunchaku.utils import get_precision

        precision = get_precision()
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors", offload=True
        )

        # Preload variants
        transformer.preload_loras(
            {
                "none": None,
                "turbo": "alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors",
            }
        )

        # Verify preloaded
        assert set(transformer.list_preloaded_loras()) == {"none", "turbo"}
        assert transformer.get_active_lora() is None

        # Switch to turbo
        transformer.switch_lora("turbo")
        assert transformer.get_active_lora() == "turbo"

        # Switch to none
        transformer.switch_lora("none")
        assert transformer.get_active_lora() == "none"

        # Clean up
        transformer.clear_preloaded_loras()
        assert len(transformer.list_preloaded_loras()) == 0

        # Cleanup
        del transformer
        gc.collect()
        torch.cuda.empty_cache()

    def test_switch_lora_produces_correct_output(self):
        """Test that switching LoRAs produces different outputs."""
        import os

        from diffusers import FluxPipeline

        from nunchaku import NunchakuFluxTransformer2dModel
        from nunchaku.utils import get_precision
        from tests.utils import compute_lpips

        precision = get_precision()
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors", offload=True
        )
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipeline.enable_sequential_cpu_offload()

        # Preload variants
        transformer.preload_loras(
            {
                "none": None,
                "turbo": "alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors",
            }
        )

        save_dir = os.path.join("test_results", "fast_lora_switching")
        os.makedirs(save_dir, exist_ok=True)

        prompt = "a beautiful sunset over mountains"
        generator = torch.Generator().manual_seed(42)

        # Generate with no LoRA
        transformer.switch_lora("none")
        image_none = pipeline(prompt, num_inference_steps=8, guidance_scale=3.5, generator=generator).images[0]
        image_none.save(os.path.join(save_dir, "none.png"))

        # Generate with turbo LoRA
        generator = torch.Generator().manual_seed(42)
        transformer.switch_lora("turbo", strength=1.0)
        image_turbo = pipeline(prompt, num_inference_steps=8, guidance_scale=3.5, generator=generator).images[0]
        image_turbo.save(os.path.join(save_dir, "turbo.png"))

        # Verify outputs are different
        lpips = compute_lpips(os.path.join(save_dir, "none.png"), os.path.join(save_dir, "turbo.png"))
        print(f"LPIPS between none and turbo: {lpips}")
        assert lpips > 0.05, "LoRA should produce different output"

        # Cleanup
        transformer.clear_preloaded_loras()
        del pipeline
        del transformer
        gc.collect()
        torch.cuda.empty_cache()

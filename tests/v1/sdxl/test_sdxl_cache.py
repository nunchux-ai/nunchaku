import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku.caching.diffusers_adapters.sdxl import apply_cache_on_pipe
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
from nunchaku.utils import get_precision, is_turing

from ...flux.utils import already_generate, compute_lpips, hash_str_to_int


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
@pytest.mark.parametrize("expected_lpips", [0.25 if get_precision() == "int4" else 0.18])
def test_sdxl_cache_lpips(expected_lpips: float):
    """Test SDXL with FBCache enabled - verify quality is maintained."""
    gc.collect()
    torch.cuda.empty_cache()

    precision = get_precision()

    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    results_dir_original = ref_root / "fp16" / "sdxl"
    results_dir_cached = ref_root / precision / "sdxl_cached"

    os.makedirs(results_dir_original, exist_ok=True)
    os.makedirs(results_dir_cached, exist_ok=True)

    prompts = [
        "Ilya Repin, Moebius, Yoshitaka Amano, 1980s nubian punk rock glam core fashion shoot, closeup, 35mm ",
        "A honeybee sitting on a flower in a garden full of yellow flowers",
        "Vibrant, tropical rainforest, teeming with wildlife, nature photography ",
        "very realistic photo of barak obama in a wing eating contest",
        "oil paint of colorful wildflowers in a meadow, Paul Signac divisionism style ",
    ]

    # Generate reference images if not exists
    if not already_generate(results_dir_original, 5):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16"
        ).to("cuda")

        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                prompt=prompt, guidance_scale=5.0, num_inference_steps=50, generator=torch.Generator().manual_seed(seed)
            ).images[0]
            result.save(os.path.join(results_dir_original, f"{seed}.png"))

        del pipeline.unet
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.vae
        del pipeline
        del result
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print(f"After original generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    # Generate images with cache enabled
    if not already_generate(results_dir_cached, 5):
        quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
            "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            unet=quantized_unet,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        )
        pipeline.unet = quantized_unet
        pipeline = pipeline.to("cuda")

        # Apply FBCache
        apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12, verbose=True)

        for prompt in prompts:
            seed = hash_str_to_int(prompt)
            result = pipeline(
                prompt=prompt, guidance_scale=5.0, num_inference_steps=50, generator=torch.Generator().manual_seed(seed)
            ).images[0]
            result.save(os.path.join(results_dir_cached, f"{seed}.png"))

        del pipeline
        del quantized_unet
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print(f"After cached generation: Free: {free/1024**2:.0f} MB  /  Total: {total/1024**2:.0f} MB")

    # Compare quality
    lpips = compute_lpips(results_dir_original, results_dir_cached)
    print(f"LPIPS (with cache): {lpips}")
    assert lpips < expected_lpips * 1.15


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
def test_sdxl_cache_functionality():
    """Test that cache hit/miss mechanism works correctly."""
    gc.collect()
    torch.cuda.empty_cache()

    quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=quantized_unet,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    # Apply FBCache with verbose to see cache hit/miss
    apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12, verbose=True)

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    print("\nGenerating with cache enabled - should see cache messages:")
    image = pipeline(
        prompt=prompt,
        guidance_scale=5.0,
        num_inference_steps=10,  # Use fewer steps for faster testing
        generator=torch.Generator().manual_seed(42),
    ).images[0]

    # Verify image was generated
    assert image is not None
    assert image.size[0] > 0 and image.size[1] > 0

    del pipeline
    del quantized_unet
    gc.collect()
    torch.cuda.empty_cache()

    print("Cache functionality test passed!")


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="Skip tests due to using Turing GPUs or FP4 precision"
)
def test_sdxl_cache_vs_no_cache():
    """Compare outputs with and without cache to ensure they match."""
    gc.collect()
    torch.cuda.empty_cache()

    quantized_unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
    )

    # First run without cache
    pipeline_no_cache = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=quantized_unet,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    prompt = "A beautiful sunset over mountains"
    seed = 123

    print("\nGenerating without cache:")
    image_no_cache = pipeline_no_cache(
        prompt=prompt, guidance_scale=5.0, num_inference_steps=10, generator=torch.Generator().manual_seed(seed)
    ).images[0]

    del pipeline_no_cache
    gc.collect()
    torch.cuda.empty_cache()

    # Second run with cache
    quantized_unet_2 = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
    )
    pipeline_cache = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=quantized_unet_2,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    apply_cache_on_pipe(pipeline_cache, residual_diff_threshold=0.12, verbose=True)

    print("\nGenerating with cache:")
    image_cache = pipeline_cache(
        prompt=prompt, guidance_scale=5.0, num_inference_steps=10, generator=torch.Generator().manual_seed(seed)
    ).images[0]

    # Save both for visual comparison
    ref_root = Path(os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref")))
    comparison_dir = ref_root / "cache_comparison"
    os.makedirs(comparison_dir, exist_ok=True)

    image_no_cache.save(comparison_dir / "no_cache.png")
    image_cache.save(comparison_dir / "with_cache.png")

    # Images should be very similar (deterministic with same seed)
    # Convert to tensors and compute difference
    import torchvision.transforms as T

    to_tensor = T.ToTensor()

    tensor_no_cache = to_tensor(image_no_cache)
    tensor_cache = to_tensor(image_cache)

    # Compute mean absolute difference
    diff = (tensor_no_cache - tensor_cache).abs().mean().item()
    print(f"\nMean absolute difference: {diff}")

    # Difference should be small (allowing for numerical variations from caching)
    # Note: Cache miss still runs all blocks, but with slight numerical differences
    assert diff < 0.02, f"Images differ too much: {diff}"

    del pipeline_cache
    del quantized_unet_2
    gc.collect()
    torch.cuda.empty_cache()

    print("Cache comparison test passed!")

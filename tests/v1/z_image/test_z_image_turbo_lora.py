import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import ZImagePipeline
from huggingface_hub import hf_hub_download

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"

model_name = "z-image-turbo-lora"
batch_size = 1
width = 1024
height = 1024
num_inference_steps = 9
guidance_scale = 0.0

ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
folder_name = f"w{width}h{height}t{num_inference_steps}g{guidance_scale}"
save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

repo_id = "Tongyi-MAI/Z-Image-Turbo"

# Default LoRA: Z-Image-Turbo-Realism-LoRA from HuggingFace
# https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA
DEFAULT_LORA_REPO = "suayptalha/Z-Image-Turbo-Realism-LoRA"
DEFAULT_LORA_FILE = "pytorch_lora_weights.safetensors"

dataset = [
    {
        "prompt": "Realism, Table Mountain, South Africa, covered in clouds on a hot, bright summers day. Use a Sony alpha 1 to capture a lot of details. use a 100mm lense. Use aperture F 1.2 to make the mountain standout. Photo taken from Blouberg Beach ",
        "negative_prompt": " ",
        "filename": "landscape",
    },
    {
        "prompt": "Realism, A futuristic tibetan god wearing ornate robes embroidered with an infinitely complex gold mandala, very old man, white beard, character concept  full body,  a weathered magical Gate with glowing runes carved into a granite cliff face, stairs lined with cherry blossom trees and jacaranda trees the entrance of goddess, ornate, beautiful, weapons, lush, nature, low angle, Protoctist style Zeng Chuanxing, widescreen, anamorphic 2 39, gold , intricate detail, hyper realistic, low angle  Symmetrical, epic scale  Cinematic, Color Grading, F 2. 8, 8K, Ultra  HD, AMOLED, Ray Tracing Global Illumination, spiritual vibes, Transparent, Translucent, Iridescent, Ray Tracing Reflections, Harris Shutter, De  Noise, VFX, SFX, anamorphic 2 39 ",
        "negative_prompt": " ",
        "filename": "art",
    },
    {
        "prompt": "Realism, 年轻的中国女子，身着红色汉服，绣工细密。妆容精致无瑕，额间点着红色花钿。发髻高盘而华丽，簪着金色凤凰头饰、红花与串珠。右手持一柄圆形折扇，扇面绘有仕女、树木与鸟。左手微抬，掌上方悬着一盏霓虹闪电形灯（⚡️），散发明亮的黄色光辉。背景是柔和灯光下的户外夜景，层叠的宝塔（西安大雁塔）成剪影状隐现，远处彩光朦胧。",
        "negative_prompt": " ",
        "filename": "portrait_chinese_prompt",
    },
]


@pytest.mark.parametrize(
    "rank,expected_lpips",
    [
        (32, {"int4-bf16": 0.5, "fp4-bf16": 0.45}),
        (128, {"int4-bf16": 0.48, "fp4-bf16": 0.42}),
        (256, {"int4-bf16": 0.46}),
    ],
)
def test_zimage_turbo_lora(rank: int, expected_lpips: dict[str, float]):
    if f"{precision}-{dtype_str}" not in expected_lpips:
        return

    # Download LoRA
    lora_path = hf_hub_download(repo_id=DEFAULT_LORA_REPO, filename=DEFAULT_LORA_FILE)

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = ZImagePipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to("cuda")
        # Apply LoRA to reference pipeline
        pipe.load_lora_weights(DEFAULT_LORA_REPO)
        run_pipeline(
            dataset=dataset,
            batch_size=1,
            pipeline=pipe,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    save_dir_nunchaku = (
        Path("test_results")
        / "nunchaku"
        / model_name
        / f"{precision}_r{rank}-{dtype_str}"
        / f"{folder_name}-bs{batch_size}"
    )
    path = f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors"
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)

    # Apply LoRA to nunchaku transformer
    transformer.update_lora_params(lora_path)

    pipe = ZImagePipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype).to("cuda")

    run_pipeline(
        dataset=dataset,
        batch_size=batch_size,
        pipeline=pipe,
        save_dir=save_dir_nunchaku,
        forward_kwargs={
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        },
    )
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips[f"{precision}-{dtype_str}"] * 1.15


def test_zimage_lora_reset():
    """Test that reset_lora() returns to original state."""
    # Load transformer
    path = f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r32-z-image-turbo.safetensors"
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)
    pipe = ZImagePipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype).to("cuda")

    save_dir = Path("test_results") / "nunchaku" / model_name / "lora_reset"
    save_dir.mkdir(parents=True, exist_ok=True)

    prompt = "A beautiful sunset over the ocean, realistic photography"
    generator = torch.Generator().manual_seed(42)

    # Generate baseline (no LoRA)
    image_before = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    image_before.save(save_dir / "before_lora.png")

    # Download and apply LoRA
    lora_path = hf_hub_download(repo_id=DEFAULT_LORA_REPO, filename=DEFAULT_LORA_FILE)
    transformer.update_lora_params(lora_path)
    transformer.set_lora_strength(1.0)

    # Reset LoRA
    transformer.reset_lora()

    # Generate after reset (should match baseline)
    generator = torch.Generator().manual_seed(42)
    image_after = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    image_after.save(save_dir / "after_reset.png")

    # Compare - should be very similar
    lpips = compute_lpips(str(save_dir / "before_lora.png"), str(save_dir / "after_reset.png"))
    print(f"LPIPS after reset: {lpips}")

    del transformer, pipe
    gc.collect()
    torch.cuda.empty_cache()

    # After reset, output should be nearly identical to baseline
    assert lpips < 0.25, f"LPIPS {lpips} too high after reset, expected < 0.25"


def test_zimage_lora_strength():
    """Test that set_lora_strength() affects output proportionally."""
    # Load transformer
    path = f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r32-z-image-turbo.safetensors"
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)
    pipe = ZImagePipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype).to("cuda")

    save_dir = Path("test_results") / "nunchaku" / model_name / "lora_strength"
    save_dir.mkdir(parents=True, exist_ok=True)

    prompt = "Realism, A beautiful mountain landscape with snow peaks"

    # Download and apply LoRA
    lora_path = hf_hub_download(repo_id=DEFAULT_LORA_REPO, filename=DEFAULT_LORA_FILE)
    transformer.update_lora_params(lora_path, strength=1.0)

    # Generate with strength 0.5
    transformer.set_lora_strength(0.5)
    assert transformer._lora_strength == 0.5, "Strength not updated correctly"

    generator = torch.Generator().manual_seed(42)
    image_half = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    image_half.save(save_dir / "strength_0.5.png")

    # Generate with strength 1.0
    transformer.set_lora_strength(1.0)
    assert transformer._lora_strength == 1.0, "Strength not updated correctly"

    generator = torch.Generator().manual_seed(42)
    image_full = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    image_full.save(save_dir / "strength_1.0.png")

    # The two images should be different (different strengths)
    lpips = compute_lpips(str(save_dir / "strength_0.5.png"), str(save_dir / "strength_1.0.png"))
    print(f"LPIPS between strength 0.5 and 1.0: {lpips}")

    del transformer, pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Different strengths should produce different outputs
    assert lpips > 0.01, f"LPIPS {lpips} too low, different strengths should produce different outputs"


def test_zimage_lora_validation():
    """Test input validation for LoRA methods."""
    # Load transformer
    path = f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r32-z-image-turbo.safetensors"
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)

    # Test 1: Negative strength in update_lora_params
    with pytest.raises(ValueError, match="non-negative"):
        transformer.update_lora_params({"dummy": torch.zeros(1)}, strength=-1.0)

    # Test 2: Empty state dict
    with pytest.raises(ValueError, match="empty"):
        transformer.update_lora_params({}, strength=1.0)

    # Test 3: Negative strength in set_lora_strength
    with pytest.raises(ValueError, match="non-negative"):
        transformer.set_lora_strength(-0.5)

    # Test 4: Valid strength values should work
    transformer.set_lora_strength(0.0)  # Zero is valid
    assert transformer._lora_strength == 0.0

    transformer.set_lora_strength(2.0)  # > 1.0 is valid
    assert transformer._lora_strength == 2.0

    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    print("All validation tests passed")

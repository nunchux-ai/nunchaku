import gc
import os

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe

from ..utils import already_generate, compute_lpips

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
PROMPT = "A cat walks on the grass, realistic style."
SEED = 42


def save_video_frames(frames, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(save_dir, f"frame_{i:04d}.png"))


def load_pipeline(gpu_id: int = 0):
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(MODEL_ID, transformer=transformer, torch_dtype=torch.float16)
    pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
    pipe.vae.enable_tiling()
    return pipe


def run_test(
    height: int = 320,
    width: int = 512,
    num_frames: int = 61,
    num_inference_steps: int = 50,
    residual_diff_threshold: float = 0.12,
    use_double_fb_cache: bool = False,
    residual_diff_threshold_multi: float | None = None,
    residual_diff_threshold_single: float | None = None,
    expected_lpips: float = 0.3,
):
    gc.collect()
    torch.cuda.empty_cache()

    folder_name = f"w{width}h{height}f{num_frames}t{num_inference_steps}"
    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    save_dir_ref = os.path.join(ref_root, "bf16", "hunyuan-video", folder_name)

    forward_kwargs = {
        "prompt": PROMPT,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
    }

    # Generate reference video (no cache)
    if not already_generate(save_dir_ref):
        pipe = load_pipeline()
        generator = torch.Generator().manual_seed(SEED)
        output = pipe(**forward_kwargs, generator=generator)
        save_video_frames(output.frames[0], save_dir_ref)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Build cache config string
    precision_str = "fbcache"
    if residual_diff_threshold > 0:
        precision_str += f"-rdt{residual_diff_threshold}"
    if use_double_fb_cache:
        precision_str += "-dfb"
    if residual_diff_threshold_multi is not None:
        precision_str += f"-rdm{residual_diff_threshold_multi}"
    if residual_diff_threshold_single is not None:
        precision_str += f"-rds{residual_diff_threshold_single}"

    save_dir_cached = os.path.join("test_results", "bf16", precision_str, "hunyuan-video", folder_name)

    # Generate cached video
    pipe = load_pipeline()
    cache_kwargs = {"residual_diff_threshold": residual_diff_threshold}
    if use_double_fb_cache:
        cache_kwargs["use_double_fb_cache"] = True
    if residual_diff_threshold_multi is not None:
        cache_kwargs["residual_diff_threshold_multi"] = residual_diff_threshold_multi
    if residual_diff_threshold_single is not None:
        cache_kwargs["residual_diff_threshold_single"] = residual_diff_threshold_single
    apply_cache_on_pipe(pipe, **cache_kwargs)

    generator = torch.Generator().manual_seed(SEED)
    output = pipe(**forward_kwargs, generator=generator)
    save_video_frames(output.frames[0], save_dir_cached)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Compare quality
    lpips = compute_lpips(save_dir_ref, save_dir_cached)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.15

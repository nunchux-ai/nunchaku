import gc
import os
import sys
from pathlib import Path

import pytest
import torch

from nunchaku.utils import get_gpu_memory, get_precision, is_turing

TESTS_DIR = Path(__file__).resolve().parents[2]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from utils import already_generate, compute_lpips
from v1.utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


class Case:
    def __init__(
        self,
        *,
        model_name: str = "Chroma1-HD",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        batch_size: int = 1,
        expected_lpips: dict[str, float] | None = None,
    ):
        self.model_name = model_name
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size

        # Allow overriding the threshold globally via env.
        env_max = os.environ.get("NUNCHAKU_CHROMA_MAX_LPIPS", "").strip()
        if env_max:
            try:
                v = float(env_max)
                expected_lpips = {
                    "int4-bf16": v,
                    "fp4-bf16": v,
                    "int4-fp16": v,
                    "fp4-fp16": v,
                }
            except Exception:
                pass

        # Conservative defaults; adjust via NUNCHAKU_CHROMA_MAX_LPIPS if needed.
        self.expected_lpips = expected_lpips or {
            "int4-bf16": 0.45,
            "fp4-bf16": 0.40,
            "int4-fp16": 0.50,
            "fp4-fp16": 0.45,
        }
        self.folder_name = f"w{width}h{height}t{num_inference_steps}g{guidance_scale}"

        self.forward_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

    def get_save_dirs(self, precision: str, dtype_str: str) -> tuple[Path, Path]:
        ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
        save_dir_16bit = Path(ref_root) / self.model_name / dtype_str / self.folder_name
        save_dir_nunchaku = (
            Path("test_results")
            / "nunchaku"
            / self.model_name
            / f"{precision}-{dtype_str}"
            / f"{self.folder_name}-bs{self.batch_size}"
        )
        return save_dir_16bit, save_dir_nunchaku


@pytest.mark.parametrize("case", [pytest.param(Case(), id="chroma1-hd")])
def test_chroma1_hd(case: Case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Chroma tests.")

    # This test requires the compiled CUDA extension.
    # Import it explicitly so we can skip with a clear message instead of failing test collection.
    try:
        import nunchaku._C  # noqa: F401
    except Exception as e:
        pytest.skip(
            "nunchaku CUDA extension is not available (missing `nunchaku._C`). "
            "Install a built wheel or build/install the project first. "
            f"Underlying error: {type(e).__name__}: {e}"
        )

    try:
        from diffusers import ChromaPipeline
    except Exception as e:
        pytest.skip(f"diffusers.ChromaPipeline is unavailable: {type(e).__name__}: {e}")

    save_dir_16bit, save_dir_nunchaku = case.get_save_dirs(precision, dtype_str)
    batch_size = case.batch_size

    # Local dir or HF repo_id for diffusers model.
    chroma_model = os.environ.get("NUNCHAKU_CHROMA_MODEL", "").strip()
    # Local path (or HF-style path) to svdq checkpoint.
    chroma_ckpt = os.environ.get("NUNCHAKU_CHROMA_SVDQ_PATH", "").strip()
    if not chroma_model:
        pytest.skip("Set NUNCHAKU_CHROMA_MODEL to your Chroma diffusers model path/repo (e.g. /path/to/Chroma1-HD).")
    if not chroma_ckpt:
        pytest.skip("Set NUNCHAKU_CHROMA_SVDQ_PATH to your SVDQ safetensors (e.g. /path/to/svdq-*_Chroma1-HD.safetensors).")

    dataset = [
        {
            "prompt": "A cat holding a sign that says hello world",
            "negative_prompt": [
                "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
            ],
            "filename": "cat_sign",
        },
        {
            "prompt": "A cinematic photo of a mountain landscape at sunrise, ultra-detailed, 35mm, natural colors",
            "negative_prompt": [
                "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
            ],
            "filename": "mountain_sunrise",
        },
    ]
    assert len(dataset) % batch_size == 0, "dataset size must be divisible by batch_size"

    # 1) Generate (and cache) 16-bit reference images.
    if not already_generate(save_dir_16bit, len(dataset)):
        pipe_ref = ChromaPipeline.from_pretrained(chroma_model, torch_dtype=torch_dtype)
        try:
            if get_gpu_memory() > 25:
                pipe_ref.enable_model_cpu_offload()
            else:
                pipe_ref.enable_sequential_cpu_offload()
        except Exception:
            pipe_ref = pipe_ref.to("cuda")
        run_pipeline(
            dataset=dataset,
            batch_size=batch_size,
            pipeline=pipe_ref,
            save_dir=save_dir_16bit,
            forward_kwargs=case.forward_kwargs,
        )
        del pipe_ref
        gc.collect()
        torch.cuda.empty_cache()

    # 2) Run with Nunchaku transformer.
    from nunchaku.models.transformers.transformer_chroma import NunchakuChromaTransformer2dModel

    transformer = NunchakuChromaTransformer2dModel.from_pretrained(chroma_ckpt, torch_dtype=torch_dtype)
    pipe = ChromaPipeline.from_pretrained(chroma_model, transformer=transformer, torch_dtype=torch_dtype)
    try:
        if get_gpu_memory() > 25:
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe = pipe.to("cuda")

    run_pipeline(
        dataset=dataset,
        batch_size=batch_size,
        pipeline=pipe,
        save_dir=save_dir_nunchaku,
        forward_kwargs=case.forward_kwargs,
    )
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 3) Quality check vs reference.
    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku, batch_size=1)
    print(f"lpips: {lpips}")
    key = f"{precision}-{dtype_str}"
    max_lpips = case.expected_lpips.get(key)
    if max_lpips is None:
        pytest.skip(f"No LPIPS threshold configured for {key}. Set NUNCHAKU_CHROMA_MAX_LPIPS to enable assertions.")
    assert lpips < max_lpips * 1.10


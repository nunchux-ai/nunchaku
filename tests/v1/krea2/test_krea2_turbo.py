import gc
import os
from pathlib import Path

import pytest
import torch
from diffusers import Krea2Pipeline

from nunchaku import NunchakuKrea2Transformer2DModel
from nunchaku.utils import get_precision, is_turing

from ...utils import already_generate, compute_lpips
from ..utils import run_pipeline

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"

model_name = "krea-2-turbo"
batch_size = 1
width = 1024
height = 1024
num_inference_steps = 8
guidance_scale = 0.0

ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
folder_name = f"w{width}h{height}t{num_inference_steps}g{guidance_scale}"
save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

repo_id = "krea/Krea-2-Turbo"

dataset = [
    {
        "prompt": "a red fox in deep snow at dusk, backlit, shallow depth of field",
        "negative_prompt": " ",
        "filename": "animal",
    },
    {
        "prompt": "a dimly lit workshop interior, single bare bulb, dust in the air, worn wooden bench",
        "negative_prompt": " ",
        "filename": "lowlight",
    },
    {
        "prompt": "portrait of an elderly woman by a window, soft daylight, fine skin texture, 85mm lens",
        "negative_prompt": " ",
        "filename": "portrait",
    },
]


@pytest.mark.skipif(is_turing(), reason="Turing GPUs do not support the fused W4A4 kernels. Skip tests.")
@pytest.mark.parametrize(
    "rank,expected_lpips",
    [
        (32, {"int4-bf16": 0.32}),
    ],
)
def test_krea2_turbo(rank: int, expected_lpips: dict[str, float]):
    if f"{precision}-{dtype_str}" not in expected_lpips:
        return

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = Krea2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to("cuda")
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
    path = f"felipesztutman/Krea-2-Turbo-W4A4-Nunchaku/svdq-{precision}_r{rank}-krea-2-turbo.safetensors"
    transformer = NunchakuKrea2Transformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)

    pipe = Krea2Pipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=torch_dtype).to("cuda")

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

import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

rank = 128  # you can also use rank=128 model to improve the quality

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
)

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", transformer=transformer, torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

# Enable cache-dit
cache_dit.enable_cache(
    pipeline,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.10,
    ),
)

image = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png")
image = image.convert("RGB")
prompt = "change the text to read '双截棍 Qwen Image Edit is here'"
inputs = {
    "image": image,
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

output = pipeline(**inputs)
output_image = output.images[0]
output_image.save(f"qwen-image-edit-cache-dit-r{rank}.png")

# Print cache statistics
cache_dit.summary(pipeline)
print(f"Image saved as qwen-image-edit-cache-dit-r{rank}.png")

import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import QwenImagePipeline

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

rank = 32  # you can also use rank=128 model to improve the quality

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image.safetensors"
)

# currently, you need to use this pipeline to offload the model to CPU
pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16)

if get_gpu_memory() > 18:
    pipe.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipe._exclude_from_cpu_offload.append("transformer")
    pipe.enable_sequential_cpu_offload()

# Enable cache-dit with config optimized for Qwen-Image
cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.10,
    ),
)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",  # for english prompt,
    "zh": "超清，4K，电影级构图",  # for chinese prompt,
}

# Generate image
prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""
negative_prompt = " "  # using an empty string if you do not have specific concept to remove

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=50,
    true_cfg_scale=4.0,
).images[0]

image.save(f"qwen-image-cache-dit-r{rank}.png")

# Print cache statistics
cache_dit.summary(pipe)
print(f"Image saved as qwen-image-cache-dit-r{rank}.png")

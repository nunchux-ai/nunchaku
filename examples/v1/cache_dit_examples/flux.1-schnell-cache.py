import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Enable cache-dit with high threshold for 4-step lightning model
# Lightning models need higher Fn/Bn and threshold
cache_dit.enable_cache(
    pipeline,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=2,
        Bn_compute_blocks=2,
        residual_diff_threshold=0.8,
        max_warmup_steps=2,
    ),
)

image = pipeline(
    "A cat holding a sign that says hello world",
    width=1024,
    height=1024,
    num_inference_steps=4,
    guidance_scale=0,
).images[0]
image.save(f"flux.1-schnell-cache-dit-{precision}.png")

# Print cache statistics
cache_dit.summary(pipeline)
print(f"Image saved as flux.1-schnell-cache-dit-{precision}.png")

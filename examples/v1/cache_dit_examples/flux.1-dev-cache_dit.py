import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxPipeline

from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

# Enable cache-dit with optimized config for FLUX.1-dev
# Please check https://github.com/vipshop/cache-dit for more details about the parameters.
cache_dit.enable_cache(
    pipeline,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.12,
    ),
)

print(f"Running FLUX.1-dev with cache-dit on GPU 0 ({precision})...")
image = pipeline(
    "A cat holding a sign that says hello world", num_inference_steps=50, generator=generator, guidance_scale=3.5
).images[0]
image.save(f"flux.1-dev-cache-dit-{precision}.png")

cache_dit.summary(pipeline)
print(f"Image saved as flux.1-dev-cache-dit-{precision}.png")

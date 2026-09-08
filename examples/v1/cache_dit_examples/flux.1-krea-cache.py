import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-krea-dev", torch_dtype=torch.bfloat16, transformer=transformer
).to("cuda")

# Enable cache-dit
cache_dit.enable_cache(
    pipeline,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.12,
    ),
)

prompt = (
    "Tiny paper origami kingdom, a river flowing through a lush valley, bright saturated image,"
    "a fox to the left, deer to the right, birds in the sky, bushes and tress all around"
)
image = pipeline(prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20).images[0]
image.save(f"flux-krea-dev-cache-dit-{precision}.png")

# Print cache statistics
cache_dit.summary(pipeline)
print(f"Image saved as flux-krea-dev-cache-dit-{precision}.png")

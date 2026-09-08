import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-fill-dev/svdq-{precision}_r32-flux.1-fill-dev.safetensors"
)
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Enable cache-dit
cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.12,
    ),
)

image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1024,
    width=1024,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
image.save(f"flux.1-fill-dev-cache-dit-{precision}.png")

# Print cache statistics
cache_dit.summary(pipe)
print(f"Image saved as flux.1-fill-dev-cache-dit-{precision}.png")

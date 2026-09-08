import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
)

pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
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

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
).convert("RGB")

prompt = "Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors"
image = pipeline(image=image, prompt=prompt, num_inference_steps=20, guidance_scale=2.5).images[0]
image.save("flux-kontext-dev-cache-dit.png")

# Print cache statistics
cache_dit.summary(pipeline)
print("Image saved as flux-kontext-dev-cache-dit.png")

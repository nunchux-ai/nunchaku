import cache_dit
import torch
from cache_dit import DBCacheConfig
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-canny-dev/svdq-{precision}_r32-flux.1-canny-dev.safetensors"
)
pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Canny-dev", transformer=transformer, torch_dtype=torch.bfloat16
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

prompt = (
    "A robot made of exotic candies and chocolates of different kinds. "
    "The background is filled with confetti and celebratory gifts."
)
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = CannyDetector()
control_image = processor(
    control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)

image = pipe(
    prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=20, guidance_scale=30.0
).images[0]

image.save(f"flux.1-canny-dev-cache-dit-{precision}.png")

# Print cache statistics
cache_dit.summary(pipe)
print(f"Image saved as flux.1-canny-dev-cache-dit-{precision}.png")

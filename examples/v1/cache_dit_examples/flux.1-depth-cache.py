import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-depth-dev/svdq-{precision}_r32-flux.1-depth-dev.safetensors"
)

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
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

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=20, guidance_scale=10.0
).images[0]
image.save(f"flux.1-depth-dev-cache-dit-{precision}.png")

cache_dit.summary(pipe)
print(f"Image saved as flux.1-depth-dev-cache-dit-{precision}.png")

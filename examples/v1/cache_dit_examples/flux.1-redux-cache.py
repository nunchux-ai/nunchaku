import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
).to("cuda")
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
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

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")
pipe_prior_output = pipe_prior_redux(image)
images = pipe(guidance_scale=2.5, num_inference_steps=20, **pipe_prior_output).images
images[0].save(f"flux.1-redux-dev-cache-dit-{precision}.png")

# Print cache statistics
cache_dit.summary(pipe)
print(f"Image saved as flux.1-redux-dev-cache-dit-{precision}.png")

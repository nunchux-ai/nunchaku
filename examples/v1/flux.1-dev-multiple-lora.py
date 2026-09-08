import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
composed_lora = compose_lora(
    [
        ("aleksa-codes/flux-ghibsky-illustration/lora.safetensors", 1),
        ("alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors", 1),
    ]
)  # set your lora strengths here when using composed lora
transformer.update_lora_params(composed_lora)
### End of LoRA Related Code ###

image = pipeline(
    "GHIBSKY style, cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
    num_inference_steps=8,
    guidance_scale=3.5,
).images[0]
image.save(f"flux.1-dev-v2-turbo-ghibsky-{precision}.png")

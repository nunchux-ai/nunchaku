import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer: NunchakuFluxTransformer2DModelV2 = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA Related Code ###
transformer.update_lora_params(
    "aleksa-codes/flux-ghibsky-illustration/lora.safetensors"
)  # Path to your LoRA safetensors, can also be a remote HuggingFace path
transformer.set_lora_strength(0.8)  # Your LoRA strength here
### End of LoRA Related Code ###

image = pipeline(
    "GHIBSKY style, cozy mountain cabin covered in snow, with smoke curling from the chimney and a warm, inviting light spilling through the windows",  # noqa: E501
    num_inference_steps=25,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(12345),
).images[0]
image.save(f"flux.1-dev-v2-ghibsky-{precision}.png")

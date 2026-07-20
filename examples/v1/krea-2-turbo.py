import torch
from diffusers import Krea2Pipeline

from nunchaku import NunchakuKrea2Transformer2DModel
from nunchaku.utils import get_precision, is_turing

if __name__ == "__main__":
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    rank = 32  # Rank of the SVDQuant low-rank branch that absorbs the outlier activations
    dtype = torch.float16 if is_turing() else torch.bfloat16  # Use float16 when Turing (20- series) GPU is used.
    transformer = NunchakuKrea2Transformer2DModel.from_pretrained(
        f"felipesztutman/Krea-2-Turbo-W4A4-Nunchaku/svdq-{precision}_r{rank}-krea-2-turbo.safetensors",
        torch_dtype=dtype,
    )

    pipe = Krea2Pipeline.from_pretrained(
        "krea/Krea-2-Turbo", transformer=transformer, torch_dtype=dtype, low_cpu_mem_usage=False
    )
    pipe.enable_sequential_cpu_offload()  # enable sequential CPU offload for low vram
    # pipe = pipe.to("cuda") # or else comment the line above and uncomment this line to put all components to GPU

    prompt = "a red fox in deep snow at dusk, backlit, shallow depth of field"

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator().manual_seed(1000),
    ).images[0]

    image.save(f"krea-2-turbo-{precision}_r{rank}_{str(dtype)}.png")

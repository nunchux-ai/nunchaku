import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku.caching.diffusers_adapters.sdxl import apply_cache_on_pipe
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

if __name__ == "__main__":
    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        "nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=unet,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    # Apply FBCache to the pipeline
    apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12, verbose=True)

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    image = pipeline(
        prompt=prompt, guidance_scale=5.0, generator=torch.Generator().manual_seed(23), num_inference_steps=50
    ).images[0]

    image.save("sdxl.png")

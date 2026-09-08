import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from huggingface_hub import hf_hub_download

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision

# Default LoRA: Z-Image-Turbo-Realism-LoRA from HuggingFace
# https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA
DEFAULT_LORA_REPO = "suayptalha/Z-Image-Turbo-Realism-LoRA"
DEFAULT_LORA_FILE = "pytorch_lora_weights.safetensors"

if __name__ == "__main__":
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    rank = 128  # Use 32 for faster sampling; 256 (INT4 only) for best quality
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors"
    )

    # Load LoRA
    lora_path = hf_hub_download(repo_id=DEFAULT_LORA_REPO, filename=DEFAULT_LORA_FILE)
    transformer.update_lora_params(lora_path)

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    ).to("cuda")

    prompt = "Realism, a young military male cooking in the kitchen for therapy"

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=8,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator().manual_seed(12345),
    ).images[0]

    image.save(f"z-image-turbo-lora-{precision}_r{rank}.png")

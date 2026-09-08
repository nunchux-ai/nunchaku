import cache_dit
import torch
from cache_dit import DBCacheConfig
from diffusers import QwenImageControlNetModel, QwenImageControlNetPipeline
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

model_name = "Qwen/Qwen-Image"
rank = 32  # you can also use rank=128 model to improve the quality

# Load components with correct dtype
controlnet = QwenImageControlNetModel.from_pretrained(
    "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch.bfloat16
)
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image.safetensors"
)

# pip install git+https://github.com/huggingface/diffusers
# Create pipeline
pipeline = QwenImageControlNetPipeline.from_pretrained(
    model_name, transformer=transformer, controlnet=controlnet, torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(True)
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

# Enable cache-dit
cache_dit.enable_cache(
    pipeline,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.10,
    ),
)

control_image = load_image("https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/depth.png")

# Generate with control
image = pipeline(
    prompt="A swanky, minimalist living room with a huge floor-to-ceiling window letting in loads of natural light. A beige couch with white cushions sits on a wooden floor, with a matching coffee table in front. The walls are a soft, warm beige, decorated with two framed botanical prints. A potted plant chills in the corner near the window. Sunlight pours through the leaves outside, casting cool shadows on the floor.",
    negative_prompt=" ",
    control_image=control_image,
    controlnet_conditioning_scale=1.0,
    num_inference_steps=30,
    true_cfg_scale=4.0,
).images[0]

# Save the result
image.save(f"qwen-image-controlnet-cache-dit-r{rank}.png")

# Print cache statistics
cache_dit.summary(pipeline)
print(f"Image saved as qwen-image-controlnet-cache-dit-r{rank}.png")

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo", subfolder="transformer", torch_dtype=torch.bfloat16
)
pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo", transformer=transformer, torch_dtype=torch.float16
).to("cuda")
apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)

output = pipeline(
    prompt="A cat walks on the grass, realistic style.",
    height=512,
    width=512,
    num_frames=61,
    num_inference_steps=50,
).frames[0]

export_to_video(output, "hunyuan-video-cache.mp4", fps=15)

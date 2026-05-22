import torch
from diffusers import ZImagePipeline
from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_gpu_memory

class ZImageTurboBackend:
    def __init__(self, model_id, optimized_model_path=None, optimized_edit_model_path=None, uma=False):
        self.model_id = model_id
        self.optimized_model_path = optimized_model_path
        self.pipeline = None
        self.uma = uma

    def load(self):
        print(f"Loading ZImageTurboBackend from {self.model_id}...")
        print(f"Loading NunchakuZImageTransformer2DModel from {self.optimized_model_path}...")
        
        # Load transformer (optimized model)
        transformer = NunchakuZImageTransformer2DModel.from_pretrained(self.optimized_model_path)

        # Load pipeline
        print("Initializing ZImagePipeline...")
        pipeline = ZImagePipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False, # standard for HF example
        )

        gpu_mem = get_gpu_memory()
        print(f"GPU memory available: {gpu_mem} GB")

        # Enable Flash Attention 2
        try:
            if hasattr(pipeline.transformer, "set_attention_backend"):
                pipeline.transformer.set_attention_backend("native")
                print("Enabled Native SDPA for Z-Image transformer")
            if hasattr(pipeline.vae, "set_attention_backend"):
                pipeline.vae.set_attention_backend("native")
                print("Enabled Native SDPA for Z-Image VAE")
        except Exception as e:
            print(f"Could not enable Flash Attention 2: {e}")

        if self.uma:
            print("UMA mode enabled: Loading all components to GPU and disabling offloads")
            pipeline.to("cuda")
        elif gpu_mem <= 18:
            print("GPU memory <= 18GB, using sequential cpu offload for low VRAM")
            # The prompt requested sequential offloading without splitting layers for Nunchaku
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()
            transformer.to("cuda")
        else:
            print("GPU memory > 18GB, using cpu offload")
            pipeline.enable_model_cpu_offload()

        self.pipeline = pipeline
        # Return twice for pipeline and edit_pipeline (though Z-Image-Turbo is T2I only)
        return self.pipeline, self.pipeline

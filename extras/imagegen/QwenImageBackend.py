import torch
from nunchaku.utils import get_gpu_memory, get_precision
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel

class QwenImageBackend:
    def __init__(self, model_id, optimized_model_path=None):
        self.model_id = model_id
        self.optimized_model_path = optimized_model_path
        self.pipeline = None
        self.rank = 32  # default rank as per example

    def load(self):
        print(f"Loading QwenImageBackend from {self.model_id}...")
        # Scheduler config (same as QwenBackend)
        import math
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Load transformer (optimized model)
        print(f"Loading NunchakuQwenImageTransformer2DModel from {self.optimized_model_path}...")
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(self.optimized_model_path)

        # Load T2I pipeline
        from diffusers import QwenImagePipeline
        pipeline = QwenImagePipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )

        # Offloading logic (same as QwenBackend)
        if get_gpu_memory() > 18:
            print("GPU memory > 18GB, using cpu offload")
            pipeline.enable_model_cpu_offload()
        else:
            print("GPU memory <= 18GB, using per-layer offloading for low VRAM")
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()

        self.pipeline = pipeline
        # For edit endpoint we reuse the same pipeline (ignores image)
        return self.pipeline, self.pipeline

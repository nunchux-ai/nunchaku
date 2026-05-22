import torch
import diffusers
try:
    from sdnq import SDNQConfig
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    SDNQ_AVAILABLE = True
except ImportError:
    print("SDNQ not found, optimized GLM loading will be skipped.")
    SDNQ_AVAILABLE = False

class GlmBackend:
    def __init__(self, model_id="Disty0/GLM-Image-SDNQ-4bit-dynamic"):
        self.model_id = model_id
        self.pipeline = None
    
    def load(self):
        print(f"Loading GLM backend from {self.model_id}...")
        
        # Load the pipeline
        # Using bfloat16 as per request snippet
        pipeline = diffusers.GlmImagePipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if SDNQ_AVAILABLE:
            # Enable INT8 MatMul for GPUs if Triton is available
            if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
                print("Applying SDNQ optimizations (INT8 MatMul)...")
                pipeline.transformer = apply_sdnq_options_to_model(pipeline.transformer, use_quantized_matmul=True)
                # pipeline.transformer = torch.compile(pipeline.transformer) # Optional, commented out as in snippet
            else:
                print("Triton or CUDA/XPU not available, skipping SDNQ optimization.")
        
        print("Enabling CPU offload for GLM pipeline...")
        pipeline.enable_model_cpu_offload()
        

        self.pipeline = pipeline
        
        # The user stated: "this one uses same pipe line for image generation and editing"
        # So we return the same pipeline for both.
        return self.pipeline, self.pipeline

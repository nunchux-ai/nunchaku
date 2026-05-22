import torch
from nunchaku.utils import get_gpu_memory, get_precision
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel

class QwenBackend:
    def __init__(self, model_id, optimized_model_path=None, optimized_edit_model_path=None, uma=False):
        self.model_id = model_id
        self.optimized_model_path = optimized_model_path
        self.optimized_edit_model_path = optimized_edit_model_path
        self.uma = uma
        self.pipeline = None
        self.rank = 32 # Default from example (was 128 in snippet, user example has 32)
        # Check snippet: rank = 32 in the example content I read.

    def load(self):
        print(f"Loading Qwen backend from {self.model_id}...")

        if not self.optimized_model_path:
             print("Warning: No optimized model path provided for QwenBackend. This requires the Nunchaku optimized model.")
        
        # Scheduler config from example
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

        # Load the base transformer (T2I)
        print(f"Loading T2I NunchakuQwenImageTransformer2DModel from {self.optimized_model_path} with FA2...")
        transformer_t2i = NunchakuQwenImageTransformer2DModel.from_pretrained(
            self.optimized_model_path,
            attn_implementation="flash_attention_2"
        )

        # Load the edit transformer
        if self.optimized_edit_model_path:
            print(f"Loading Edit NunchakuQwenImageTransformer2DModel from {self.optimized_edit_model_path} with FA2...")
            transformer_edit = NunchakuQwenImageTransformer2DModel.from_pretrained(
                self.optimized_edit_model_path,
                attn_implementation="flash_attention_2"
            )
        else:
            print(f"Using shared transformer for Edit pipeline...")
            transformer_edit = transformer_t2i

        print(f"Loading QwenImagePipeline from {self.model_id}...")
        # Use QwenImagePipeline (T2I)
        from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline
        
        text_encoder = None
        if self.uma:
            print("UMA mode: Loading text_encoder in 8-bit using BitsAndBytes...")
            from transformers import BitsAndBytesConfig, AutoModel
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder = AutoModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # 1. Load Edit Pipeline (To handle processor correctly)
        print(f"Loading QwenImageEditPlusPipeline from {self.model_id}...")
        
        pipeline_kwargs = {
            "transformer": transformer_edit,
            "scheduler": scheduler,
            "torch_dtype": torch.bfloat16
        }
        if text_encoder is not None:
            pipeline_kwargs["text_encoder"] = text_encoder

        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_id, 
            **pipeline_kwargs
        )

        # 2. Create T2I Pipeline sharing components (except transformer if separate)
        print("Creating QwenImagePipeline (T2I) with shared components...")
        
        # Ensure we have a text_encoder and tokenizer
        if edit_pipeline.text_encoder is None:
            print("Text encoder not found in edit_pipeline, loading manually...")
            # Load from model_id or subfolder
            if text_encoder is None:
                from transformers import AutoModel
                text_encoder = AutoModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, trust_remote_code=True)
            
            # CRITICAL FIX: Assign it back to the pipeline!
            edit_pipeline.register_modules(text_encoder=text_encoder)
        else:
            text_encoder = edit_pipeline.text_encoder

        tokenizer = edit_pipeline.tokenizer
        
        if tokenizer is None:
             print("Tokenizer not found in edit_pipeline, loading manually...")
             from transformers import AutoTokenizer
             tokenizer = AutoTokenizer.from_pretrained(self.model_id, subfolder="tokenizer", trust_remote_code=True)
             edit_pipeline.register_modules(tokenizer=tokenizer)

        pipeline = QwenImagePipeline(
            transformer=transformer_t2i,
            scheduler=edit_pipeline.scheduler,
            vae=edit_pipeline.vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        
        # Manually assign processors if needed (though QwenImagePipeline creates its own image_processor)
        # pipeline.feature_extractor = edit_pipeline.image_processor

        # Logic for offloading / UMA
        if self.uma:
            print("UMA mode enabled: Text encoder loaded in 8-bit. Moving other components to GPU.")
            # Note: 8-bit text encoder is already handled by bitsandbytes (on GPU or offloaded as needed, typically GPU).
            
            # Explicitly move transformers to CUDA
            print("Moving T2I Transformer to CUDA...")
            transformer_t2i.to("cuda")
            
            if transformer_edit != transformer_t2i:
                print("Moving Edit Transformer to CUDA...")
                transformer_edit.to("cuda")

            # We need to ensure other components (VAE) are on CUDA.
            if hasattr(edit_pipeline, "vae") and edit_pipeline.vae:
                print("Moving VAE to CUDA...")
                edit_pipeline.vae.to("cuda")
            
            # Since we can't call pipeline.to("cuda") generally if 8-bit modules are present (sometimes safe, sometimes not),
            # we manually handle it or trust loaded components.
            pass
            # Note: pipeline (T2I) shares components, so it should be on cuda too.
        else:
            print("Non-UMA mode: Using aggressive per-layer offloading.")
            transformer_t2i.set_offload(
                True, use_pin_memory=True, num_blocks_on_gpu=8
            )
            if self.optimized_edit_model_path:
                transformer_edit.set_offload(
                    True, use_pin_memory=True, num_blocks_on_gpu=8
                )
            
            edit_pipeline._exclude_from_cpu_offload.append("transformer")
            edit_pipeline.enable_sequential_cpu_offload()
            
            # The T2I pipeline (pipeline) also needs to handle offloading.
            # If we manually loaded text_encoder, it might not be attached to edit_pipeline's offload hooks.
            # We should enable sequential CPU offload for the T2I pipeline too.
            pipeline.enable_sequential_cpu_offload()

            if self.optimized_edit_model_path:
                pass
        
        self.pipeline = pipeline
        self.edit_pipeline = edit_pipeline
        return self.pipeline, self.edit_pipeline

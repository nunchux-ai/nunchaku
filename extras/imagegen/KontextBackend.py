import torch
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import FluxKontextPipeline

class KontextBackend:
    def __init__(self, model_id, optimized_model_path=None):
        self.model_id = model_id
        self.optimized_model_path = optimized_model_path
        self.pipeline = None

    def load(self):
        print(f"Loading Kontext backend from {self.model_id}...")
        
        if self.optimized_model_path:
            print(f"Loading optimized transformer from {self.optimized_model_path}...")
            # Load the optimized transformer (Nunchaku style! *hyah!*)
            try:
                from nunchaku import NunchakuFluxTransformer2dModel
            except ImportError:
                 print("Oops, nunchaku not found! Please install it for optimized magic.")
                 raise

            transformer = NunchakuFluxTransformer2dModel.from_pretrained(self.optimized_model_path)

            text_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            text_encoder_2_4bit = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                quantization_config=text_quant_config,
                torch_dtype=torch.bfloat16  # bfloat16 for your NVIDIA setup—faster magic!
            )
            
            # Load the pipeline with the optimized transformer
            # We need FluxKontextPipeline for editing magic!
            pipeline = FluxKontextPipeline.from_pretrained(
                self.model_id,
                text_encoder_2=text_encoder_2_4bit,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            print("No optimized model path provided for KontextBackend. Falling back to standard loading if possible, or maybe we should insist on one?")
            # Original code implied usage of optimized model for Kontext was the main path, but let's support standard if needed,
            # or minimally just load standard logic if that was the fallback. 
            # Looking at original code: "if args.optimized_model: ... else: ... Flux2Pipeline" 
            # Wait, the original code fell back to Flux2Pipeline if no optimized model was present!
            # The user request says: "create KontextBackend.py that creates a pipeline from base and optional optimized paths"
            # So KontextBackend *should* support both optimized and unoptimized? Or was the fallback in original code actually switching to Flux2?
            # Original code: 
            # if args.optimized_model: 
            #    # Load Nunchaku stuff 
            #    pipeline = FluxKontextPipeline(...)
            # else:
            #    # Load standard stuff
            #    pipeline = Flux2Pipeline(...) 
            #
            # The USER request says: "KontextBackend.py that creates a pipeline from base and optional optimized paths".
            # This implies if I choose "kontext" backend but don't provide optimized path, it should still load a FluxKontextPipeline (presumably unoptimized/standard).
            # However, FluxKontextPipeline might expect specific components.
            # Let's assume standard loading for FluxKontextPipeline if no optimized model is separate.
            
            print(f"Loading standard FluxKontextPipeline from {self.model_id}...")
            # Assuming standard 4-bit loading for memory savings similar to before
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Use basic from_pretrained
            pipeline = FluxKontextPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16
                # We might need quantization for components if memory is tight, but from_pretrained handles a lot.
                # Let's keep it simple for now as we don't have the Nunchaku specific loading here.
            )
            # Actually, if we look at how specialized the optimized loading was, standard loading might just be:
            # pipeline = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=...)
            
        self.pipeline = pipeline
        self.pipeline.to("cuda")
        
        # Additional setup if needed (like offload)
        # self.pipeline.enable_model_cpu_offload() # User code had this for optimized path
        
        return self.pipeline, self.pipeline

import torch
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor, BitsAndBytesConfig
from diffusers import Flux2Pipeline, AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

class Flux2Backend:
    def __init__(self, model_id):
        self.model_id = model_id
        self.pipeline = None

    def load(self):
        print(f"Loading Flux2 backend from {self.model_id}...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, 
            subfolder="scheduler",
            torch_dtype=torch.bfloat16
        )

        # VAE - loaded manually with full precision
        vae = AutoencoderKLFlux2.from_pretrained(
            self.model_id, 
            subfolder="vae", 
            torch_dtype=torch.float16
        )

        tokenizer = PixtralProcessor.from_pretrained(
            self.model_id, 
            subfolder="tokenizer", 
            torch_dtype=torch.float16
        )

        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_id, 
            subfolder="text_encoder", 
            torch_dtype=torch.float16, 
            quantization_config=quantization_config
        )
        
        dit = Flux2Transformer2DModel.from_pretrained(
            self.model_id, 
            subfolder="transformer", 
            torch_dtype=torch.float16, 
            quantization_config=quantization_config
        )


        # Standard loading without Nunchaku optimization
        # Constructing pipeline manually rather than from_pretrained
        pipeline = Flux2Pipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=dit,
        )
        
        self.pipeline = pipeline
        self.pipeline.to("cuda")
        self.pipeline.transformer.set_attention_backend("flash")
        
        return self.pipeline, self.pipeline

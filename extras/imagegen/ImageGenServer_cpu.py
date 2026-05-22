import argparse
import base64
import io
import time
import torch
import uvicorn
import numpy as np
import gc
import asyncio
from fastapi import FastAPI, HTTPException, Request
from accelerate import infer_auto_device_map, dispatch_model
from pydantic import BaseModel
from diffusers import (
    Flux2Pipeline,
    Flux2Transformer2DModel,
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from transformers import Mistral3ForConditionalGeneration, AutoProcessor

# Argument parsing
parser = argparse.ArgumentParser(description="Flux2 Image Generation Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev", help="Path or Repo ID of the model")
args = parser.parse_args()

app = FastAPI()

# Global components
text_encoder = None
tokenizer = None
transformer = None
vae = None
scheduler = None
image_processor = None
request_lock = asyncio.Lock()

# Device maps
text_encoder_map = None
transformer_map = None
vae_map = None

GPU_MEMORY_FRACTION = 0.90

def load_model():
    global text_encoder, tokenizer, transformer, vae, scheduler, image_processor
    global text_encoder_map, transformer_map, vae_map
    
    print(f"Loading model from {args.model}...")
    
    try:
        print("Loading Flux2 components...")

        # Calculate max memory per GPU
        #max_memory = {}
        #if torch.cuda.is_available():
        #    for i in range(torch.cuda.device_count()):
        #        total_mem = torch.cuda.get_device_properties(i).total_memory
        #        max_memory[i] = int(total_mem * GPU_MEMORY_FRACTION)

        max_memory = {
          0: "5GB",   # leave a little headroom
 #        1: "10GB",
          "cpu": "120GB"  # your 128GB RAM minus OS
        }
        
        # Load Text Encoder (Mistral3) on CPU
        print("Loading Text Encoder on CPU...")
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            args.model,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print("Calculating Text Encoder device map...")
        text_encoder_map = infer_auto_device_map(text_encoder, max_memory=max_memory)

        # Load Tokenizer on CPU
        print("Loading Tokenizer on CPU...")
        tokenizer = AutoProcessor.from_pretrained(
            args.model,
            subfolder="tokenizer",
            device_map="cpu"
        )

        # Load Transformer on CPU
        print("Loading Transformer on CPU...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print("Calculating Transformer device map...")
        transformer_map = infer_auto_device_map(transformer, max_memory=max_memory)

        # Load VAE on CPU
        print("Loading VAE on CPU...")
        vae = AutoencoderKLFlux2.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print("Calculating VAE device map...")
        vae_map = infer_auto_device_map(vae, max_memory=max_memory)

        # Initialize Scheduler
        print("Initializing Scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model,
            subfolder="scheduler"
        )

        # Initialize Image Processor
        print("Initializing Image Processor...")
        # VAE scale factor logic from pipeline
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    
    print("Model loaded successfully!")

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    quality: str = "standard"
    style: str = "vivid"

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/v1/images/generations")
async def generate_image(request: ImageGenerationRequest):
    if not transformer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    async with request_lock:
        print(f"Received request: {request.prompt}")

        # Parse size
        try:
            width, height = map(int, request.size.split("x"))
        except ValueError:
            width, height = 1024, 1024

        num_inference_steps = 28
        guidance_scale = 4.0
        max_sequence_length = 512
        device = torch.device("cuda")
        dtype = torch.bfloat16

        images = []
        
        # 1. Generate embeddings on CPU
        print("Generating embeddings...")
        flush()
        prompt_embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=request.prompt,            
#            device=torch.device("cpu"),
            max_sequence_length=max_sequence_length
        )


#       prompt_embeds = prompt_embeds.to("cuda")       
     
        # 2. Prepare Latents
        # Flux latents are turned into 2x2 patches and packed. 
        # This means the latent width and height has to be divisible by the patch size.
        # So the vae scale factor is multiplied by the patch size to account for this
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        
        height = height or 1024
        width = width or 1024
        
        # Resize to be divisible by vae_scale_factor * 2
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        num_channels_latents = transformer.config.in_channels // 4
        shape = (1, num_channels_latents * 4, height // 2, width // 2)
        
        # 3. Prepare IDs
        # We need to prepare text_ids and latent_ids
        # prompt_embeds shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = prompt_embeds.shape
        
        # Repeat for num_images_per_prompt (assuming 1 for now per loop iteration)
        # If request.n > 1, we loop outside or handle batching. Here we loop outside.
        
        # Prepare text IDs
        text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(device)
        
        for _ in range(request.n):
            # Generate random latents
            latents = torch.randn(shape, device=device, dtype=dtype)
            
            # Prepare latent IDs
            latent_ids = Flux2Pipeline._prepare_latent_ids(latents).to(device)
            
            # Pack latents
            packed_latents = Flux2Pipeline._pack_latents(latents)
            
            # 4. Prepare Timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = packed_latents.shape[1]
            mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            
            # --- SWAP TRANSFORMER TO CUDA ---
            print("Moving Transformer to CUDA...")
            flush()
            dispatch_model(transformer, device_map=transformer_map)
            
            # 5. Denoising Loop
            print("Starting denoising loop on CUDA...")
            scheduler.set_begin_index(0)
            
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(packed_latents.shape[0])
            
            for i, t in enumerate(timesteps):
                start_time = time.time()
                # broadcast to batch dimension
                timestep = t.expand(packed_latents.shape[0]).to(packed_latents.dtype)
                
                noise_pred = transformer(
                    hidden_states=packed_latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

                # step
                packed_latents = scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
                
                step_time = time.time() - start_time
                print(f"Step {i+1}/{num_inference_steps}: {step_time:.2f}s")

            # --- SWAP TRANSFORMER TO CPU ---
            print("Moving Transformer to CPU...")
            transformer.to("cpu")
            flush()

            # --- SWAP VAE TO CUDA ---
            print("Moving VAE to CUDA...")
            dispatch_model(vae, device_map=vae_map)

            # 6. Decode
            print("Decoding on CUDA...")
            # Move packed_latents to CUDA for decoding (already there, but ensuring)
            packed_latents = packed_latents.to(device)
            latent_ids = latent_ids.to(device)
            
            latents = Flux2Pipeline._unpack_latents_with_ids(packed_latents, latent_ids)
            
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean
            latents = Flux2Pipeline._unpatchify_latents(latents)

            image = vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")[0]
            
            # --- SWAP VAE TO CPU ---
            print("Moving VAE to CPU...")
            vae.to("cpu")
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images.append({"b64_json": img_str})

        return {
            "created": int(time.time()),
            "data": images
        }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

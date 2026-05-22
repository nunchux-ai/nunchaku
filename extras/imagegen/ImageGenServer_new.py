import argparse
import base64
import io
import time
import torch
import uvicorn
import gc
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import FluxPipeline, FluxKontextPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from PIL import Image

# Argument parsing
parser = argparse.ArgumentParser(description="Flux Image Generation Server with Nunchaku")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev", help="Path or Repo ID of the base model")
parser.add_argument("--optimized-model", type=str, required=True, help="Path to the optimized Nunchaku model safetensors file")
args = parser.parse_args()

app = FastAPI()

# Global components
pipeline = None
img2img_pipeline = None
request_lock = asyncio.Lock()

def load_model():
    global pipeline, img2img_pipeline
    
    print(f"Loading base model from {args.model}...")
    print(f"Loading optimized transformer from {args.optimized_model}...")
    
    try:
        # Load the optimized transformer
        # Ensuring transformer is in bfloat16 to match the pipeline expectation
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(args.optimized_model)
        
        # Load the pipeline with the optimized transformer
        pipeline = FluxPipeline.from_pretrained(
            args.model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        # Load the Img2Img/Context pipeline sharing the same components
        # We use strict component sharing to avoid VRAM duplication
        print("Initializing FluxKontextPipeline for image inputs...")
        # Since FluxKontextPipeline shares architecture with FluxPipeline, we can initialize it with the same components
        img2img_pipeline = FluxKontextPipeline.from_pretrained(
            args.model,
            transformer=pipeline.transformer,
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            scheduler=pipeline.scheduler,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        # Enable CPU offload for the main pipeline. 
        # Since components are shared, this should handle memory management for both.
        pipeline.enable_model_cpu_offload()
        # img2img_pipeline.enable_model_cpu_offload() # Avoid double hook registration
        
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
    image: Optional[str] = None # Base64 encoded image

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/v1/images/generations")
async def generate_image(request: ImageGenerationRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")

    async with request_lock:
        print(f"Received request: {request.prompt}")

        # Parse size
        try:
            width, height = map(int, request.size.split("x"))
        except ValueError:
            width, height = 1024, 1024

        # Flux requires dimensions to be multiples of 16 (or 8 depending on VAE)
        # Standard Flux dev usually works well with 1024x1024
        # We'll ensure they are divisible by 16 just in case
        width = (width // 16) * 16
        height = (height // 16) * 16

        images = []
        
        try:
            input_image = None
            if request.image:
                try:
                    # Handle data URI if present
                    img_data = request.image
                    if "," in img_data:
                        img_data = img_data.split(",")[1]
                    
                    input_bytes = base64.b64decode(img_data)
                    input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
                    # Resize input image to match request size
                    input_image = input_image.resize((width, height), Image.LANCZOS)
                    print(f"Processed input image of size {input_image.size}")
                except Exception as e:
                    print(f"Failed to decode input image: {e}")
                    raise HTTPException(status_code=400, detail="Invalid image data")

            # Generate images
            if input_image:
                # Use FluxKontextPipeline
                print("Running FluxKontextPipeline...")
                generated_images = pipeline(
                    image=input_image,
                    prompt=request.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=28, 
                    guidance_scale=2.5, # Recommended for Kontext
                    num_images_per_prompt=request.n
                ).images
            else:
                # Use standard FluxPipeline
                print("Running FluxPipeline...")
                generated_images = pipeline(
                    request.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=28, # Standard for Flux Dev
                    guidance_scale=3.5,     # Nunchaku example uses 3.5
                    num_images_per_prompt=request.n
                ).images
            
            for image in generated_images:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({"b64_json": img_str})
                
        except Exception as e:
            print(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            flush()

        return {
            "created": int(time.time()),
            "data": images
        }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

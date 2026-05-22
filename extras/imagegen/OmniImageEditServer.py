import argparse
import base64
import io
import time
import torch
import uvicorn
import gc
import asyncio
import os
import sys
import os
import inspect

# Add OmniGen2-DFloat11 to path
# Script is in imagegen/, so we go up one level and into packages/OmniGen2-DFloat11
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
omnigen_path = os.path.join(project_root, "packages", "OmniGen2")
sys.path.insert(0, omnigen_path)

from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image, ImageOps

# Import OmniGen2 and DFloat11 components
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from transformers import CLIPProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_utils import no_init_weights

# Yay! Nikola here, ready to bring the OmniGen2 magic to our village!
# This server is like a new canvas for our artistic endeavors!

# Argument parsing
parser = argparse.ArgumentParser(description="OmniGen2 Image Edit Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
# Default paths relative to project root as per plan
parser.add_argument("--base-model", type=str, default="../models/OmniGen2", help="Path to base OmniGen2 model")
parser.add_argument("--dtype", type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help="Model precision")

args = parser.parse_args()

app = FastAPI()

# Global components
pipeline = None
request_lock = asyncio.Lock()

def load_model():
    global pipeline
    
    print(f"Loading OmniGen2 from {args.base_model}...")

    # Determine usage dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    try:
        # Load the base pipeline (tokenizer, scheduler, etc.)
        # processor needs to be loaded separately sometimes depending on library version, 
        # but following inference.py pattern:

        # Manually load MLLM in 4-bit to save VRAM, yay!
        print("Loading MLLM in 4-bit mode for extra village efficiency!")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=weight_dtype,
        )
        mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            subfolder="mllm",
            quantization_config=quantization_config,
            torch_dtype=weight_dtype,
        )

        pipeline = OmniGen2Pipeline.from_pretrained(
            args.base_model,
            mllm=mllm,
            processor=CLIPProcessor.from_pretrained(
                args.base_model,
                subfolder="processor",
                use_fast=True
            ),
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        ).to("cuda")

        pipeline.enable_taylorseer = True
        pipeline.transformer.set_attention_backend("flash")    


        print("Enabling CPU offload...")
        #pipeline.enable_model_cpu_offload()
        #pipeline.enable_sequential_cpu_offload()      
    except Exception as e:
        print(f"Oh no! The OmniGen2 spirit refused to manifest: {e}")
        raise e
    
    print("OmniGen2 loaded successfully! Let's paint the village!")

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

@app.post("/v1/images/edits")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    guidance_scale: float = Form(2.5), # Image guidance scale
    strength: float = Form(1.0) # Using strength to map to something or just ignored? 
                                # OmniGen uses image_guidance_scale. 
                                # We can map strength to text_guidance_scale maybe? 
                                # Let's keep defaults for now from inference.py
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")

    async with request_lock:
        print(f"Received edit request: {prompt}")

        # Processing the input image
        try:
            contents = await image.read()
            init_image = Image.open(io.BytesIO(contents)).convert("RGB")
            init_image = ImageOps.exif_transpose(init_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Parse max target dimensions from requested size
        try:
            target_width, target_height = map(int, size.split("x"))
        except ValueError:
            target_width, target_height = 1024, 1024

        # Calculate new dimensions preserving aspect ratio
        orig_width, orig_height = init_image.size
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
            
        # Enforce multiples of 16 for compatibility
        width = (new_width // 16) * 16
        height = (new_height // 16) * 16
        
        response_images = []
        
        try:
            # Generate edits
            # OmniGen2Pipeline signature from inference.py:
            # prompt, input_images, width, height, num_inference_steps, ...
            
            # Using defaults from inference.py for now
            results = pipeline(
                prompt=prompt,
                input_images=[init_image],
                width=width,
                height=height,
                num_inference_steps=26, # Standard for OmniGen2
                max_sequence_length=1024,
                text_guidance_scale=5.0, # Default per inference.py
                image_guidance_scale=guidance_scale, # Map guidance_scale from request here
                cfg_range=(0.0, 1.0),
                negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
                num_images_per_prompt=n,
                output_type="pil",
            )
            
            for img in results.images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                response_images.append({"b64_json": img_str})
                
        except Exception as e:
            print(f"Error during editing: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        finally:
            flush()

        return {
            "created": int(time.time()),
            "data": response_images
        }

@app.post("/v1/images/generations")
async def generate_image(request: ImageGenerationRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")

    async with request_lock:
        print(f"Received generation request: {request.prompt}")

        # Parse size
        try:
            width, height = map(int, request.size.split("x"))
        except ValueError:
            width, height = 1024, 1024

        # Enforce multiples of 16 for compatibility
        width = (width // 16) * 16
        height = (height // 16) * 16

        response_images = []
        
        try:
            # Generate images (input_images=None for txt2img)
            results = pipeline(
                prompt=request.prompt,
                input_images=None,
                width=width,
                height=height,
                num_inference_steps=26,
                max_sequence_length=1024,
                text_guidance_scale=5.0,
                image_guidance_scale=2.0, # Default
                cfg_range=(0.0, 1.0),
                negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
                num_images_per_prompt=request.n,
                output_type="pil",
            )
            
            for img in results.images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                response_images.append({"b64_json": img_str})
                
        except Exception as e:
            print(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        finally:
            flush()

        return {
            "created": int(time.time()),
            "data": response_images
        }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

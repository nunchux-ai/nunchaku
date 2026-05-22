import argparse
import base64
import io
import time
import torch
import uvicorn
import gc
import asyncio
import traceback
from typing import List, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image, ImageOps

# Argument parsing
parser = argparse.ArgumentParser(description="Flux Image Edit Server with Nunchaku")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="Path or Repo ID of the base model")
parser.add_argument("--optimized-model", type=str, default=None, help="Path to the optimized Nunchaku model safetensors file")
parser.add_argument("--optimized-edit-model", type=str, default=None, help="Path to the optimized Nunchaku model safetensors file for editing (optional)")
parser.add_argument("--backend", type=str, default="kontext", choices=["kontext", "flux2", "qwen", "glm", "zimage"], help="Backend to use: 'kontext', 'flux2', 'qwen', 'glm', or 'zimage'")
parser.add_argument("--steps", type=int, default=28, help="Default number of inference steps")
parser.add_argument("--guidance-scale", type=float, default=3.5, help="Default guidance scale")
parser.add_argument("--qwenimage", action="store_true", help="Use QwenImageBackend (T2I only) instead of full Qwen edit backend")
parser.add_argument("--uma", action="store_true", help="Enable Unified Memory Architecture mode (load all to GPU, disable offload)")
args = parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    load_model()
    yield
    # Shutdown logic (if any) could go here

app = FastAPI(lifespan=lifespan)

# Global components
IMAGE_DIMENSION_ALIGNMENT = 32
pipeline = None
edit_pipeline = None
request_lock = asyncio.Lock()
is_sleeping_flag = False
sleep_requested = False

def load_model():
    global pipeline, edit_pipeline
    
    try:
        if args.backend == "kontext":
            import KontextBackend
            print(f"Initializing KontextBackend...")
            backend = KontextBackend.KontextBackend(args.model, args.optimized_model)
            pipeline, edit_pipeline = backend.load()
        elif args.backend == "flux2":
            import Flux2Backend
            print(f"Initializing Flux2Backend...")
            backend = Flux2Backend.Flux2Backend(args.model)
            pipeline, edit_pipeline = backend.load()
        elif args.backend == "glm":
            import GlmBackend
            print(f"Initializing GlmBackend...")
            # Use provided model or default to the one in the snippet if args.model is generic
            # The user might pass the specific GLM model via --model, or we default in GlmBackend.
            # Let's pass args.model if it's not the default flux one, otherwise let GlmBackend use its default.
            model_to_use = args.model if args.model != "black-forest-labs/FLUX.1-Kontext-dev" else "Disty0/GLM-Image-SDNQ-4bit-dynamic"
            backend = GlmBackend.GlmBackend(model_to_use)
            pipeline, edit_pipeline = backend.load()
        elif args.backend.startswith("qwen"):
            if args.qwenimage:
                import QwenImageBackend
                print(f"Initializing QwenImageBackend (T2I only)...")
                backend = QwenImageBackend.QwenImageBackend(args.model, args.optimized_model)
                pipeline, edit_pipeline = backend.load()
            else:
                import QwenBackend
                print(f"Initializing QwenBackend...")
                backend = QwenBackend.QwenBackend(args.model, args.optimized_model, optimized_edit_model_path=args.optimized_edit_model, uma=args.uma)
                pipeline, edit_pipeline = backend.load()
        elif args.backend == "zimage":
            import ZImageTurboBackend
            print(f"Initializing ZImageTurboBackend...")
            backend = ZImageTurboBackend.ZImageTurboBackend(args.model, args.optimized_model, uma=args.uma)
            pipeline, edit_pipeline = backend.load()
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
            
    except Exception as e:
        print(f"Oh no! The model refused to wake up: {e}")
        raise e
    
    # Enable progress bar for diffusers
    import diffusers.utils.logging
    diffusers.utils.logging.enable_progress_bar()
    diffusers.utils.logging.set_verbosity_info()
    
    print("Model loaded successfully! Ready for editing quests!")

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
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


@app.post("/v1/sleep")
async def sleep_endpoint():
    global is_sleeping_flag, sleep_requested
    sleep_requested = True
    try:
        async with request_lock:
            if not is_sleeping_flag and sleep_requested:
                print("Sleep requested, moving models to CPU...")
                for p in [pipeline, edit_pipeline]:
                    if not p: continue
                    for name, component in p.components.items():
                        if isinstance(component, torch.nn.Module):
                            # Special handling for Nunchaku which blocks .to() if offload is True
                            if hasattr(component, "set_offload") and getattr(component, "offload", False):
                                component.set_offload(False)
                                component._nunchaku_was_offloaded = True

                            try:
                                component.to("cpu")
                            except Exception as e:
                                pass
                flush()
                is_sleeping_flag = True
    finally:
        sleep_requested = False
    return {"status": "sleep completed", "is_sleeping": is_sleeping_flag}

@app.post("/v1/wake_up")
async def wake_up_endpoint():
    global is_sleeping_flag, sleep_requested
    sleep_requested = False
    async with request_lock:
        if is_sleeping_flag:
            print("Waking up, restoring models to CUDA...")
            for p in [pipeline, edit_pipeline]:
                if not p: continue
                excluded = getattr(p, "_exclude_from_cpu_offload", [])
                for name, component in p.components.items():
                    if isinstance(component, torch.nn.Module):
                        if getattr(component, "_nunchaku_was_offloaded", False):
                            component.set_offload(True, use_pin_memory=True, num_blocks_on_gpu=8)
                            for attr in ["img_in", "txt_in", "txt_norm", "time_text_embed", "norm_out", "proj_out"]:
                                if hasattr(component, attr):
                                    try:
                                        getattr(component, attr).to("cuda")
                                    except Exception:
                                        pass
                            component._nunchaku_was_offloaded = False
                        elif not hasattr(component, "_hf_hook") or name in excluded:
                            try:
                                component.to("cuda")
                            except Exception:
                                pass
            is_sleeping_flag = False
    return {"status": "awoken", "is_sleeping": False}

@app.get("/v1/is_sleeping")
async def is_sleeping_endpoint():
    return {"is_sleeping": is_sleeping_flag}

@app.post("/v1/images/edits")
async def edit_image(
    image: Union[List[UploadFile], UploadFile] = File(...),
    prompt: str = Form(...),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"), # Default to b64_json
    guidance_scale: Optional[float] = Form(None), 
    num_inference_steps: Optional[int] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    seed: Optional[int] = Form(None)
):
    # Use CLI defaults if not provided
    steps = num_inference_steps if num_inference_steps is not None else args.steps
    cfg_scale = guidance_scale if guidance_scale is not None else args.guidance_scale
    neg_prompt = negative_prompt if negative_prompt is not None else "" # Default empty for now, or maybe None?
    
    generator = None
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    print(f"Using seed: {seed}")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if not edit_pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if sleep_requested or is_sleeping_flag:
        raise HTTPException(status_code=503, detail="Server is sleeping or trying to sleep.")

    async with request_lock:
        print(f"Received edit request: {prompt}")

        # Processing the input image(s)
        input_files = image if isinstance(image, list) else [image]
        init_images = []

        try:
            for img_file in input_files:
                await img_file.seek(0)
                contents = await img_file.read()
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                init_images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
        if not init_images:
             raise HTTPException(status_code=400, detail="No images provided")

        # Parse max target dimensions from requested size
        try:
            target_width, target_height = map(int, size.split("x"))
        except ValueError:
            target_width, target_height = 1024, 1024

        # Calculate new dimensions preserving aspect ratio based on the first image
        first_image = init_images[0]
        orig_width, orig_height = first_image.size
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Ensure dimensions are aligned to 32 for compatibility (e.g. GLM-Image)
        width = (new_width // IMAGE_DIMENSION_ALIGNMENT) * IMAGE_DIMENSION_ALIGNMENT
        height = (new_height // IMAGE_DIMENSION_ALIGNMENT) * IMAGE_DIMENSION_ALIGNMENT
        
        # Resize input images to match the calculated target size, padding if necessary
        resized_images = []
        for img in init_images:
            if img.size != (width, height):
                # Use ImageOps.pad to preserve aspect ratio and center in the target size
                # This handles cases where subsequent images might have different ARs
                img = ImageOps.pad(img, (width, height), method=Image.LANCZOS, color=(0, 0, 0))
            resized_images.append(img)
            
        # If single image, pass as item, if multiple, pass as list
        # GLM pipeline has a bug where it checks len() on the input, so it must be a list
        if len(resized_images) > 1 or args.backend == "glm":
            image_input = resized_images
        else:
            image_input = resized_images[0]

        response_images = []
        
        try:
            if args.backend.startswith("qwen"):
                # Qwen specific parameters
                # guidance_scale maps to true_cfg_scale
                if args.qwenimage: # QwenImageBackend is T2I only, so it doesn't take an image
                    generated_images = edit_pipeline(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        true_cfg_scale=cfg_scale,
                        num_images_per_prompt=n,
                        generator=generator,
                    ).images
                else: # Full Qwen edit backend takes an image (or list of images now)
                    generated_images = edit_pipeline(
                        image=image_input,
                        prompt=prompt,
                        height=height,
                        width=width,
                        negative_prompt=neg_prompt,
                        num_inference_steps=steps,
                        true_cfg_scale=cfg_scale,
                        num_images_per_prompt=n,
                        generator=generator,
                    ).images
            else:
                # Standard Flux/Kontext or GLM
                # GLM I2I Fix: Manually move vision encoder to GPU because get_image_features escapes hooks
                if args.backend == "glm" and hasattr(edit_pipeline, "vision_language_encoder"):
                     print("Manually moving GLM Vision Encoder to GPU...")
                     edit_pipeline.vision_language_encoder.to("cuda")
                
                try:
                    generated_images = edit_pipeline(
                        image=image_input,
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps, 
                        guidance_scale=cfg_scale,
                        num_images_per_prompt=n,
                        generator=generator,
                    ).images
                finally:
                    if args.backend == "glm" and hasattr(edit_pipeline, "vision_language_encoder"):
                         print("Moving GLM Vision Encoder back to CPU...")
                         edit_pipeline.vision_language_encoder.to("cpu")
            
            for img in generated_images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                if response_format == "b64_json":
                    response_images.append({"b64_json": img_str})
                else:
                    # If url is requested we can't really do it without storage, so we fallback or error?
                    # For now, let's just assume simple b64_json as per request
                    response_images.append({"b64_json": img_str}) # Fallback
                
        except Exception as e:
            print(f"Error during editing: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
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

    if sleep_requested or is_sleeping_flag:
        raise HTTPException(status_code=503, detail="Server is sleeping or trying to sleep.")

    async with request_lock:
        #print(f"Received generation request: {request.prompt}")

        # Parse size
        try:
            width, height = map(int, request.size.split("x"))
        except ValueError:
            width, height = 1024, 1024

        # Ensure dimensions are aligned to 32
        width = (width // IMAGE_DIMENSION_ALIGNMENT) * IMAGE_DIMENSION_ALIGNMENT
        height = (height // IMAGE_DIMENSION_ALIGNMENT) * IMAGE_DIMENSION_ALIGNMENT

        response_images = []
        
        try:
            # Generate images (no image argument for txt2img!)
            steps = request.num_inference_steps if request.num_inference_steps is not None else args.steps
            cfg_scale = request.guidance_scale if request.guidance_scale is not None else args.guidance_scale
            # negative_prompt not in standard request body in original snippet, but we added it to model
            neg_prompt = request.negative_prompt if request.negative_prompt is not None else ""
            
            generator = None
            import random
            seed = request.seed
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            
            print(f"Using seed: {seed}")
            generator = torch.Generator(device="cuda").manual_seed(seed)

            if args.backend.startswith("qwen"):
                 generated_images = pipeline(
                    prompt=request.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps, 
                    true_cfg_scale=cfg_scale,     
                    num_images_per_prompt=request.n,
                    negative_prompt=neg_prompt,
                    generator=generator,
                ).images
            else:
                generated_images = pipeline(
                    prompt=request.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps, 
                    guidance_scale=cfg_scale,     
                    num_images_per_prompt=request.n,
                    generator=generator,
                    # Not passing negative_prompt here for generation unless we confirm support in standard Flux pipeline?
                ).images
            
            for img in generated_images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                response_images.append({"b64_json": img_str})
                
        except Exception as e:
            print(f"Error during generation: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            flush()

        return {
            "created": int(time.time()),
            "data": response_images
        }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

# Changed from https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py
import os
import random
import time
from datetime import datetime

import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from image_gen_aux import DepthPreprocessor
from PIL import Image
from vars import (
    DEFAULT_GUIDANCE_CANNY,
    DEFAULT_GUIDANCE_DEPTH,
    DEFAULT_INFERENCE_STEP_CANNY,
    DEFAULT_INFERENCE_STEP_DEPTH,
    DEFAULT_STYLE_NAME,
    EXAMPLES,
    HEIGHT,
    MAX_SEED,
    STYLE_NAMES,
    STYLES,
    WIDTH,
)

from nunchaku.models.safety_checker import SafetyChecker
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from utils import get_args

# import gradio last to avoid conflicts with other imports
import gradio as gr  # noqa: isort: skip

args = get_args()

pipeline_class = None
processor = None
model_name = None

model_name = f"{args.model}-dev"
pipeline_class = FluxControlPipeline
if args.model == "canny":
    processor = CannyDetector()
else:
    assert args.model == "depth", f"Model {args.model} not supported"
    processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")

if args.precision == "bf16":
    pipeline = pipeline_class.from_pretrained(
        f"black-forest-labs/FLUX.1-{model_name.capitalize()}", torch_dtype=torch.bfloat16
    )
    pipeline = pipeline.to("cuda")
    pipeline.precision = "bf16"
else:
    assert args.precision in ["int4", "fp4"]
    pipeline_init_kwargs = {}
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-{model_name}/svdq-{args.precision}_r32-flux.1-{model_name}.safetensors"
    )
    pipeline_init_kwargs["transformer"] = transformer
    if args.use_qencoder:
        from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel

        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
            "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
        )
        pipeline_init_kwargs["text_encoder_2"] = text_encoder_2

    pipeline = pipeline_class.from_pretrained(
        f"black-forest-labs/FLUX.1-{model_name.capitalize()}", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
    )
    pipeline = pipeline.to("cuda")
    pipeline.precision = args.precision

safety_checker = SafetyChecker("cuda", disabled=args.no_safety_checker)


def run(
    image, prompt: str, style: str, prompt_template: str, num_inference_steps: int, guidance_scale: float, seed: int
) -> tuple[Image, str]:
    if args.model == "canny":
        processed_img = processor(image["composite"]).convert("RGB")
    else:
        assert args.model == "depth"
        processed_img = processor(image["composite"])[0].convert("RGB")

    is_unsafe_prompt = False
    if not safety_checker(prompt):
        is_unsafe_prompt = True
        prompt = "A peaceful world."
    prompt = prompt_template.format(prompt=prompt)
    print(f"Prompt: {prompt}")
    start_time = time.time()
    result_image = pipeline(
        prompt=prompt,
        control_image=processed_img,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]

    latency = time.time() - start_time
    if latency < 1:
        latency = latency * 1000
        latency_str = f"{latency:.2f}ms"
    else:
        latency_str = f"{latency:.2f}s"
    if is_unsafe_prompt:
        latency_str += " (Unsafe prompt detected)"
    torch.cuda.empty_cache()
    if args.count_use:
        if os.path.exists(f"{args.model}-use_count.txt"):
            with open(f"{args.model}-use_count.txt", "r") as f:
                count = int(f.read())
        else:
            count = 0
        count += 1
        current_time = datetime.now()
        print(f"{current_time}: {count}")
        with open(f"{args.model}-use_count.txt", "w") as f:
            f.write(str(count))
        with open(f"{args.model}-use_record.txt", "a") as f:
            f.write(f"{current_time}: {count}\n")
    return result_image, latency_str


with gr.Blocks(css_paths="assets/style.css", title=f"SVDQuant Flux.1-{model_name} Demo") as demo:
    with open("assets/description.html", "r") as f:
        DESCRIPTION = f.read()
    # Get the GPU properties
    if torch.cuda.device_count() > 0:
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_memory = gpu_properties.total_memory / (1024**3)  # Convert to GiB
        gpu_name = torch.cuda.get_device_name(0)
        device_info = f"Running on {gpu_name} with {gpu_memory:.0f} GiB memory."
    else:
        device_info = "Running on CPU 🥶 This demo does not work on CPU."
    notice = '<strong>Notice:</strong>&nbsp;We will replace unsafe prompts with a default prompt: "A peaceful world."'

    def get_header_str():

        if args.count_use:
            if os.path.exists(f"{args.model}-use_count.txt"):
                with open(f"{args.model}-use_count.txt", "r") as f:
                    count = int(f.read())
            else:
                count = 0
            count_info = (
                f"<div style='display: flex; justify-content: center; align-items: center; text-align: center;'>"
                f"<span style='font-size: 18px; font-weight: bold;'>Total inference runs: </span>"
                f"<span style='font-size: 18px; color:red; font-weight: bold;'>&nbsp;{count}</span></div>"
            )
        else:
            count_info = ""
        header_str = DESCRIPTION.format(
            precision=args.precision,
            model_name=args.model,
            device_info=device_info,
            notice=notice,
            count_info=count_info,
        )
        return header_str

    header = gr.HTML(get_header_str())
    demo.load(fn=get_header_str, outputs=header)

    with gr.Row(elem_id="main_row"):
        with gr.Column(elem_id="column_input"):
            gr.Markdown("## INPUT", elem_id="input_header")
            with gr.Group():
                canvas = gr.ImageEditor(
                    height=640,
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    type="pil",
                    label="Input",
                    show_label=False,
                    show_download_button=True,
                    interactive=True,
                    transforms=[],
                    canvas_size=(1024, 1024),
                    scale=1,
                    format="png",
                    layers=False,
                )
                with gr.Row():
                    prompt = gr.Text(label="Prompt", placeholder="Enter your prompt", scale=6)
                    run_button = gr.Button("Run", scale=1, elem_id="run_button")
            with gr.Row():
                style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME, scale=1)
                prompt_template = gr.Textbox(
                    label="Prompt Style Template", value=STYLES[DEFAULT_STYLE_NAME], scale=2, max_lines=1
                )

            with gr.Row():
                seed = gr.Slider(label="Seed", show_label=True, minimum=0, maximum=MAX_SEED, value=233, step=1, scale=4)
                randomize_seed = gr.Button("Random Seed", scale=1, min_width=50, elem_id="random_seed")
            with gr.Accordion("Advanced options", open=False):
                with gr.Group():
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=DEFAULT_INFERENCE_STEP_CANNY if args.model == "canny" else DEFAULT_INFERENCE_STEP_DEPTH,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=DEFAULT_GUIDANCE_CANNY if args.model == "canny" else DEFAULT_GUIDANCE_DEPTH,
                    )

        with gr.Column(elem_id="column_output"):
            gr.Markdown("## OUTPUT", elem_id="output_header")
            with gr.Group():
                result = gr.Image(
                    format="png",
                    height=640,
                    image_mode="RGB",
                    type="pil",
                    label="Result",
                    show_label=False,
                    show_download_button=True,
                    interactive=False,
                    elem_id="output_image",
                )
                latency_result = gr.Text(label="Inference Latency", show_label=True)

            gr.Markdown("### Instructions")
            gr.Markdown("**1**. Enter a text prompt (e.g., a cat)")
            gr.Markdown("**2**. Upload or paste an image")
            gr.Markdown("**3**. Change the image style using a style template")
            gr.Markdown("**4**. Adjust the effect of sketch guidance using the slider")
            gr.Markdown("**5**. Try different seeds to generate different results")

    run_inputs = [canvas, prompt, style, prompt_template, num_inference_steps, guidance_scale, seed]
    run_outputs = [result, latency_result]

    gr.Examples(examples=EXAMPLES[args.model], inputs=run_inputs, outputs=run_outputs, fn=run)

    randomize_seed.click(
        lambda: random.randint(0, MAX_SEED),
        inputs=[],
        outputs=seed,
        api_name=False,
        queue=False,
    ).then(run, inputs=run_inputs, outputs=run_outputs, api_name=False)

    style.change(
        lambda x: STYLES[x],
        inputs=[style],
        outputs=[prompt_template],
        api_name=False,
        queue=False,
    )
    gr.on(
        triggers=[prompt.submit, run_button.click],
        fn=run,
        inputs=run_inputs,
        outputs=run_outputs,
        api_name=False,
    )

    gr.Markdown("MIT Accessibility: https://accessibility.mit.edu/", elem_id="accessibility")


if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True, root_path=args.gradio_root_path)

# Changed from https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py
import os
import random
import time
from datetime import datetime

import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from PIL import Image
from vars import DEFAULT_GUIDANCE, DEFAULT_INFERENCE_STEP, EXAMPLES, MAX_SEED

from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from utils import get_args

# import gradio last to avoid conflicts with other imports
import gradio as gr  # noqa: isort: skip

args = get_args()

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
).to("cuda")

if args.precision == "bf16":
    pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipeline = pipeline.to("cuda")
    pipeline.precision = "bf16"
else:
    assert args.precision in ["int4", "fp4"]
    pipeline_init_kwargs = {}
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{args.precision}_r32-flux.1-dev.safetensors"
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder=None,
        text_encoder_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to("cuda")
    pipeline.precision = args.precision


def run(image, num_inference_steps: int, guidance_scale: float, seed: int) -> tuple[Image, str]:
    pipe_prior_output = pipe_prior_redux(image["composite"])

    start_time = time.time()
    result_image = pipeline(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        **pipe_prior_output,
    ).images[0]

    latency = time.time() - start_time
    if latency < 1:
        latency = latency * 1000
        latency_str = f"{latency:.2f}ms"
    else:
        latency_str = f"{latency:.2f}s"

    torch.cuda.empty_cache()
    if args.count_use:
        if os.path.exists("use_count.txt"):
            with open("use_count.txt", "r") as f:
                count = int(f.read())
        else:
            count = 0
        count += 1
        current_time = datetime.now()
        print(f"{current_time}: {count}")
        with open("use_count.txt", "w") as f:
            f.write(str(count))
        with open("use_record.txt", "a") as f:
            f.write(f"{current_time}: {count}\n")
    return result_image, latency_str


with gr.Blocks(css_paths="assets/style.css", title="SVDQuant Flux.1-redux-dev Demo") as demo:
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

    def get_header_str():

        if args.count_use:
            if os.path.exists("use_count.txt"):
                with open("use_count.txt", "r") as f:
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
        header_str = DESCRIPTION.format(device_info=device_info, count_info=count_info)
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
                    run_button = gr.Button("Run", elem_id="run_button")

            with gr.Row():
                seed = gr.Slider(
                    label="Seed",
                    show_label=True,
                    minimum=0,
                    maximum=MAX_SEED,
                    value=233,
                    step=1,
                    scale=4,
                )
                randomize_seed = gr.Button("Random Seed", scale=1, min_width=50, elem_id="random_seed")
            with gr.Accordion("Advanced options", open=False):
                with gr.Group():
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=DEFAULT_INFERENCE_STEP,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        value=DEFAULT_GUIDANCE,
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
            gr.Markdown("**1**. Upload or paste an image")
            gr.Markdown(
                "**2**. Adjust the effect of sketch guidance and inference steps using sliders under Advanced options"
            )
            gr.Markdown("**3**. Try different seeds to generate different results")

    run_inputs = [canvas, num_inference_steps, guidance_scale, seed]
    run_outputs = [result, latency_result]

    gr.Examples(examples=EXAMPLES, inputs=run_inputs, outputs=run_outputs, fn=run)

    randomize_seed.click(
        lambda: random.randint(0, MAX_SEED),
        inputs=[],
        outputs=seed,
        api_name=False,
        queue=False,
    ).then(run, inputs=run_inputs, outputs=run_outputs, api_name=False)

    gr.on(triggers=[run_button.click], fn=run, inputs=run_inputs, outputs=run_outputs, api_name=False)

    gr.Markdown("MIT Accessibility: https://accessibility.mit.edu/", elem_id="accessibility")


if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True, root_path=args.gradio_root_path)

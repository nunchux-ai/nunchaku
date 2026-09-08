# Changed from https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py
import argparse
import os
import random
import time
from datetime import datetime

import GPUtil
import spaces
import torch
from vars import EXAMPLES, MAX_SEED

from nunchaku.models.safety_checker import SafetyChecker
from utils import get_pipeline

# import gradio last to avoid conflicts with other imports
import gradio as gr  # noqa: isort: skip


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--precisions",
        type=str,
        default=["int4"],
        nargs="*",
        choices=["int4", "bf16"],
        help="Which precisions to use",
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker")
    parser.add_argument("--count-use", action="store_true", help="Whether to count the number of uses")
    parser.add_argument("--gradio-root-path", type=str, default="")
    return parser.parse_args()


args = get_args()


pipelines = []
pipeline_init_kwargs = {}
for i, precision in enumerate(args.precisions):

    pipeline = get_pipeline(
        precision=precision,
        use_qencoder=args.use_qencoder,
        device="cuda",
        pipeline_init_kwargs={**pipeline_init_kwargs},
    )
    pipelines.append(pipeline)
    if i == 0:
        pipeline_init_kwargs["vae"] = pipeline.vae
        pipeline_init_kwargs["text_encoder"] = pipeline.text_encoder

safety_checker = SafetyChecker("cuda", disabled=args.no_safety_checker)


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    pag_scale: float = 0,
    seed: int = 0,
):
    print(f"Prompt: {prompt}")
    is_unsafe_prompt = False
    if not safety_checker(prompt):
        is_unsafe_prompt = True
        prompt = "A peaceful world."
    images, latency_strs = [], []
    for i, pipeline in enumerate(pipelines):
        gr.Progress(track_tqdm=True)
        start_time = time.time()
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            pag_scale=pag_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
        end_time = time.time()
        latency = end_time - start_time
        if latency < 1:
            latency = latency * 1000
            latency_str = f"{latency:.2f}ms"
        else:
            latency_str = f"{latency:.2f}s"
        images.append(image)
        latency_strs.append(latency_str)
    if is_unsafe_prompt:
        for i in range(len(latency_strs)):
            latency_strs[i] += " (Unsafe prompt detected)"
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

    return *images, *latency_strs


with open("./assets/description.html", "r") as f:
    DESCRIPTION = f.read()
gpus = GPUtil.getGPUs()
if len(gpus) > 0:
    gpu = gpus[0]
    memory = gpu.memoryTotal / 1024
    device_info = f"Running on {gpu.name} with {memory:.0f} GiB memory."
else:
    device_info = "Running on CPU 🥶 This demo does not work on CPU."
notice = '<strong>Notice:</strong>&nbsp;We will replace unsafe prompts with a default prompt: "A peaceful world."'

with gr.Blocks(
    css_paths=[f"assets/frame{len(args.precisions)}.css", "assets/common.css"],
    title="SVDQuant SANA-1600M Demo",
) as demo:

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
        header_str = DESCRIPTION.format(device_info=device_info, notice=notice, count_info=count_info)
        return header_str

    header = gr.HTML(get_header_str())
    demo.load(fn=get_header_str, outputs=header)

    with gr.Row():
        image_results, latency_results = [], []
        for i, precision in enumerate(args.precisions):
            with gr.Column():
                gr.Markdown(f"# {precision.upper()}", elem_id="image_header")
                with gr.Group():
                    image_result = gr.Image(
                        format="png",
                        image_mode="RGB",
                        label="Result",
                        show_label=False,
                        show_download_button=True,
                        interactive=False,
                    )
                    latency_result = gr.Text(label="Inference Latency", show_label=True)
                    image_results.append(image_result)
                    latency_results.append(latency_result)
    with gr.Row():
        prompt = gr.Text(
            label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False, scale=4
        )
        run_button = gr.Button("Run", scale=1)

    with gr.Row():
        seed = gr.Slider(label="Seed", show_label=True, minimum=0, maximum=MAX_SEED, value=233, step=1, scale=4)
        randomize_seed = gr.Button("Random Seed", scale=1, min_width=50, elem_id="random_seed")
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            height = gr.Slider(label="Height", minimum=256, maximum=4096, step=32, value=1024)
            width = gr.Slider(label="Width", minimum=256, maximum=4096, step=32, value=1024)
        with gr.Group():
            num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=50, step=1, value=20)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, step=0.1, value=5)
            pag_scale = gr.Slider(label="PAG Scale", minimum=0, maximum=10, step=0.1, value=2.0)

    input_args = [prompt, height, width, num_inference_steps, guidance_scale, pag_scale, seed]

    gr.Examples(examples=EXAMPLES, inputs=input_args, outputs=[*image_results, *latency_results], fn=generate)

    gr.on(
        triggers=[prompt.submit, run_button.click],
        fn=generate,
        inputs=input_args,
        outputs=[*image_results, *latency_results],
        api_name=False,
    )
    randomize_seed.click(
        lambda: random.randint(0, MAX_SEED), inputs=[], outputs=seed, api_name=False, queue=False
    ).then(fn=generate, inputs=input_args, outputs=[*image_results, *latency_results], api_name=False, queue=False)

    gr.Markdown("MIT Accessibility: https://accessibility.mit.edu/", elem_id="accessibility")


if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", debug=True, share=True, root_path=args.gradio_root_path)

import argparse
import os

import torch
from vars import PROMPT_TEMPLATES

from utils import get_pipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="schnell", choices=["schnell", "dev"], help="Which FLUX.1 model to use"
    )
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "fp4", "bf16"], help="Which precision to use"
    )
    parser.add_argument(
        "--prompt", type=str, default="A cat holding a sign that says hello world", help="Prompt for the image"
    )
    parser.add_argument("--seed", type=int, default=2333, help="Random seed (-1 for random)")
    parser.add_argument("-t", "--num-inference-steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("-o", "--output-path", type=str, default="output.png", help="Image output path")
    parser.add_argument("-g", "--guidance-scale", type=float, default=0, help="Guidance scale.")
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    known_args, _ = parser.parse_known_args()

    if known_args.model == "dev":
        parser.set_defaults(num_inference_steps=50, guidance_scale=3.5)
        parser.add_argument(
            "-n",
            "--lora-name",
            type=str,
            default="None",
            choices=PROMPT_TEMPLATES.keys(),
            help="Name of the LoRA layer",
        )
        parser.add_argument("-a", "--lora-weight", type=float, default=1, help="Weight of the LoRA layer")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pipeline = get_pipeline(
        model_name=args.model,
        precision=args.precision,
        use_qencoder=args.use_qencoder,
        lora_name=getattr(args, "lora_name", "None"),
        lora_weight=getattr(args, "lora_weight", 1),
        device="cuda",
    )

    if args.model == "dev":
        prompt = PROMPT_TEMPLATES[args.lora_name].format(prompt=args.prompt)
    else:
        prompt = args.prompt

    image = pipeline(
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator().manual_seed(args.seed) if args.seed >= 0 else None,
    ).images[0]
    output_dir = os.path.dirname(os.path.abspath(os.path.expanduser(args.output_path)))
    os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_path)


if __name__ == "__main__":
    main()

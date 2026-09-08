import argparse
import os

import torch

from utils import get_pipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "bf16"], help="Which precision to use"
    )
    parser.add_argument(
        "--prompt", type=str, default="A cat holding a sign that says hello world", help="Prompt for the image"
    )
    parser.add_argument("--seed", type=int, default=2333, help="Random seed (-1 for random)")
    parser.add_argument("-t", "--num-inference-steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("-o", "--output-path", type=str, default="output.png", help="Image output path")
    parser.add_argument("-g", "--guidance-scale", type=float, default=5, help="Guidance scale.")
    parser.add_argument("--pag-scale", type=float, default=2.0, help="PAG scale")
    parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    known_args, _ = parser.parse_known_args()

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pipeline = get_pipeline(precision=args.precision, use_qencoder=args.use_qencoder, device="cuda")

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

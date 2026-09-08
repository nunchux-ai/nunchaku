import argparse
import time

import torch
from torch import nn
from tqdm import trange

from utils import get_pipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "bf16"], help="Which precision to use"
    )

    parser.add_argument("-t", "--num-inference-steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("-g", "--guidance-scale", type=float, default=5, help="Guidance scale")
    parser.add_argument("--pag-scale", type=float, default=2.0, help="PAG scale")
    parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    parser.add_argument("--width", type=int, default=1024, help="Width of the image")

    # Test related
    parser.add_argument("--warmup-times", type=int, default=2, help="Number of warmup times")
    parser.add_argument("--test-times", type=int, default=10, help="Number of test times")
    parser.add_argument(
        "--mode",
        type=str,
        default="end2end",
        choices=["end2end", "step"],
        help="Measure mode: end-to-end latency or per-step latency",
    )
    parser.add_argument(
        "--ignore_ratio", type=float, default=0.2, help="Ignored ratio of the slowest and fastest steps"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pipeline = get_pipeline(precision=args.precision, device="cuda")

    dummy_prompt = "A cat holding a sign that says hello world"

    latency_list = []
    if args.mode == "end2end":
        pipeline.set_progress_bar_config(position=1, desc="Step", leave=False)
        for _ in trange(args.warmup_times, desc="Warmup", position=0, leave=False):
            pipeline(
                prompt=dummy_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            torch.cuda.synchronize()
        for _ in trange(args.test_times, desc="Warmup", position=0, leave=False):
            start_time = time.time()
            pipeline(
                prompt=dummy_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            latency_list.append(end_time - start_time)
    elif args.mode == "step":
        inputs = {}

        def get_input_hook(module: nn.Module, input_args, input_kwargs):
            inputs["args"] = input_args
            inputs["kwargs"] = input_kwargs

        pipeline.transformer.register_forward_pre_hook(get_input_hook, with_kwargs=True)

        pipeline(
            prompt=dummy_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=1,
            guidance_scale=args.guidance_scale,
            output_type="latent",
        )

        for _ in trange(args.warmup_times, desc="Warmup", position=0, leave=False):
            pipeline.transformer(*inputs["args"], **inputs["kwargs"])
            torch.cuda.synchronize()
        for _ in trange(args.test_times, desc="Warmup", position=0, leave=False):
            start_time = time.time()
            pipeline.transformer(*inputs["args"], **inputs["kwargs"])
            torch.cuda.synchronize()
            end_time = time.time()
            latency_list.append(end_time - start_time)

    latency_list = sorted(latency_list)
    ignored_count = int(args.ignore_ratio * len(latency_list) / 2)
    if ignored_count > 0:
        latency_list = latency_list[ignored_count:-ignored_count]

    print(f"Latency: {sum(latency_list) / len(latency_list):.5f} s")


if __name__ == "__main__":
    main()

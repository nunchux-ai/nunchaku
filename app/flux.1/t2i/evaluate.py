import argparse
import os

import torch
from data import get_dataset
from tqdm import tqdm

from utils import get_pipeline, hash_str_to_int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="schnell",
        choices=["schnell", "schnell_v2", "dev"],
        help="Which FLUX.1 model to use",
    )
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "fp4", "bf16"], help="Which precision to use"
    )
    parser.add_argument(
        "-d", "--datasets", type=str, nargs="*", default=["MJHQ", "DCI"], help="The benchmark datasets to evaluate on."
    )
    parser.add_argument("-t", "--num-inference-steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("-g", "--guidance-scale", type=float, default=0, help="Guidance scale.")
    parser.add_argument("-o", "--output-root", type=str, default=None, help="Image output path")
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=1,
        help="You will generate images for the subset specified by [chunk-start::chunk-step].",
    )
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="You will generate images for the subset specified by [chunk-start::chunk-step].",
    )
    parser.add_argument(
        "--max-dataset-size", type=int, default=5000, help="Maximum number of images to generate for each dataset"
    )
    known_args, _ = parser.parse_known_args()

    if known_args.model == "dev":
        parser.set_defaults(num_inference_steps=50, guidance_scale=3.5)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    assert args.chunk_step > 0
    assert 0 <= args.chunk_start < args.chunk_step

    pipeline = get_pipeline(model_name=args.model, precision=args.precision, device="cuda")
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)

    output_root = args.output_root
    if output_root is None:
        output_root = f"results/{args.model}/{args.precision}/"

    for dataset_name in args.datasets:
        output_dirname = os.path.join(output_root, dataset_name)
        os.makedirs(output_dirname, exist_ok=True)
        dataset = get_dataset(name=dataset_name, max_dataset_size=args.max_dataset_size)
        if args.chunk_step > 1:
            dataset = dataset.select(range(args.chunk_start, len(dataset), args.chunk_step))
        for row in tqdm(dataset):
            filename = row["filename"]
            prompt = row["prompt"]
            seed = hash_str_to_int(filename)
            image = pipeline(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).images[0]
            image.save(os.path.join(output_dirname, f"{filename}.png"))


if __name__ == "__main__":
    main()

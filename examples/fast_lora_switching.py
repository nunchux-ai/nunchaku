"""
Example: Fast LoRA Switching

This example demonstrates how to use the fast LoRA switching feature
for real-time interactive applications.

使用方法:
    python examples/fast_lora_switching.py

输出:
    - test_results/fast_lora_switching/ 目录下的图片
    - 终端输出性能对比
"""

import os
import time

import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision


def main():
    precision = get_precision()
    print(f"Using precision: {precision}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    output_dir = "test_results/fast_lora_switching"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{precision}_r32-flux.1-krea-dev.safetensors"
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Define LoRAs to use
    loras = {
        "base": None,
        "anime": "XLabs-AI/flux-lora-collection/anime_lora.safetensors",
        "realism": "XLabs-AI/flux-lora-collection/realism_lora.safetensors",
    }

    # Preload LoRA variants
    print("\nPreloading LoRA variants...")
    t0 = time.perf_counter()
    transformer.preload_loras(loras)
    preload_time = time.perf_counter() - t0
    print(f"Preload time: {preload_time * 1000:.1f}ms")
    print(f"Available variants: {transformer.list_preloaded_loras()}")

    # Benchmark fast switching
    print("\n" + "=" * 60)
    print("Benchmarking Fast LoRA Switching")
    print("=" * 60)
    n_switches = 20
    lora_names = list(loras.keys())

    # Warm up
    for name in lora_names:
        transformer.switch_lora(name)
    torch.cuda.synchronize()

    # Benchmark
    switch_times = []
    for i in range(n_switches):
        name = lora_names[i % len(lora_names)]
        t0 = time.perf_counter()
        transformer.switch_lora(name)
        torch.cuda.synchronize()
        switch_times.append(time.perf_counter() - t0)

    avg_switch_time = sum(switch_times) / len(switch_times)
    print(
        f"Fast switch - Avg: {avg_switch_time * 1000:.2f}ms, "
        f"Min: {min(switch_times) * 1000:.2f}ms, "
        f"Max: {max(switch_times) * 1000:.2f}ms"
    )

    # Benchmark traditional switching
    print("\n" + "=" * 60)
    print("Benchmarking Traditional LoRA Switching")
    print("=" * 60)

    # Reset first
    transformer.reset_lora()

    traditional_times = []
    for i in range(6):  # 2 cycles through 3 LoRAs
        name = lora_names[i % len(lora_names)]
        lora_path = loras[name]

        transformer.reset_lora()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        if lora_path is not None:
            transformer.update_lora_params(lora_path)
        torch.cuda.synchronize()
        traditional_times.append(time.perf_counter() - t0)

    avg_traditional_time = sum(traditional_times) / len(traditional_times)
    print(
        f"Traditional switch - Avg: {avg_traditional_time * 1000:.2f}ms, "
        f"Min: {min(traditional_times) * 1000:.2f}ms, "
        f"Max: {max(traditional_times) * 1000:.2f}ms"
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    speedup = avg_traditional_time / avg_switch_time
    print(f"Fast switching:        {avg_switch_time * 1000:.2f}ms")
    print(f"Traditional switching: {avg_traditional_time * 1000:.2f}ms")
    print(f"Speedup:               {speedup:.1f}x faster")
    print(f"Preload cost:          {preload_time * 1000:.1f}ms (one-time)")
    break_even = preload_time / (avg_traditional_time - avg_switch_time)
    print(f"Break-even:            {break_even:.1f} switches")

    # Generate images with each LoRA
    print("\n" + "=" * 60)
    print("Generating Images")
    print("=" * 60)

    prompts = {
        "base": "a beautiful sunset over mountains, highly detailed, 8k",
        "anime": "a beautiful sunset over mountains, anime style, studio ghibli",
        "realism": "a beautiful sunset over mountains, photorealistic, DSLR photo",
    }

    # Reset to use fast switching
    transformer.reset_lora()
    transformer.preload_loras(loras)

    for name in lora_names:
        print(f"\nGenerating with '{name}' LoRA...")
        transformer.switch_lora(name)

        t0 = time.perf_counter()
        image = pipeline(
            prompts[name],
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        gen_time = time.perf_counter() - t0

        output_path = os.path.join(output_dir, f"{name}.png")
        image.save(output_path)
        print(f"  Saved: {output_path} ({gen_time:.2f}s)")

    print("\n" + "=" * 60)
    print(f"All images saved to: {output_dir}/")
    print("=" * 60)

    # Cleanup
    transformer.clear_preloaded_loras()


if __name__ == "__main__":
    main()

<div align="center" id="nunchaku_logo">
  <img src="https://www.nunchux.ai/Logo.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>Paper</b></a> | <a href="https://nunchaku.tech/docs/nunchaku/"><b>Docs</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>Website</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>Blog</b></a> | <a href="https://nunchux.ai/"><b>Demo</b></a> | <a href="https://huggingface.co/nunchaku-ai"><b>Hugging Face</b></a> | <a href="https://modelscope.cn/organization/nunchaku-tech"><b>ModelScope</b></a> | <a href="https://github.com/nunchux-ai/ComfyUI-nunchaku"><b>ComfyUI</b></a>
</h3>
<div align="center">
  <a href="https://trendshift.io/repositories/20447" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20447" alt="nunchux-ai/nunchaku | Trendshift" style="width: 120px; height: 26px;" width="120" height="26"/></a>
  <a href=https://discord.gg/Wk6PnwX9Sm target="_blank"><img src=https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FWk6PnwX9Sm%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total height=22px></a>
  <a href=https://huggingface.co/datasets/nunchaku-ai/cdn/resolve/main/nunchaku/assets/wechat.jpg target="_blank"><img src=https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white height=22px></a>
  <a href=https://deepwiki.com/nunchux-ai/nunchaku target="_blank"><img src=https://deepwiki.com/badge.svg height=22px></a>
</div>

**Nunchaku** is a high-performance inference engine optimized for 4-bit neural networks, as introduced in our paper [SVDQuant](http://arxiv.org/abs/2411.05007). For the underlying quantization library, check out [DeepCompressor](https://github.com/nunchux-ai/deepcompressor).

Join our user groups on [**Discord**](https://discord.gg/Wk6PnwX9Sm) and [**WeChat**](https://huggingface.co/datasets/nunchaku-ai/cdn/resolve/main/nunchaku/assets/wechat.jpg) to engage in discussions with the community! More details can be found [here](https://github.com/nunchux-ai/nunchaku/issues/149). If you have any questions, run into issues, or are interested in contributing, don’t hesitate to reach out!

## News

- **[2026-01-12]** 🚀 **v1.2.0 Released!** Enjoy a **20–30% Z-Image performance boost**, seamless **LoRA support with native ComfyUI nodes**, **INT4 support for 20-series GPUs**, and **compatibility with ComfyUI 0.7**!
- **[2025-12-20]** 🔥 Release **4-bit [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)**! Download on [Hugging Face](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo) or [ModelScope](https://modelscope.cn/models/nunchaku-tech/nunchaku-z-image-turbo), and try it with our [example script](examples/v1/z-image-turbo.py).
- **[2025-09-25]** 🔥 Release **4-bit [4/8-step lightning Qwen-Image-Edit-2509](https://huggingface.co/lightx2v/Qwen-Image-Lightning)**! Download on [Hugging Face](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509) or [ModelScope](https://modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509), and try it with our [example script](examples/v1/qwen-image-edit-2509-lightning.py).
- **[2025-09-24]** 🔥 Released **4-bit [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)**! Models are available on [Hugging Face](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509). Try them out with our [example script](examples/v1/qwen-image-edit-2509.py). Lightning models will follow up!

<details>
<summary>More</summary>

- **[2025-09-09]** 🔥 Released [**4-bit Qwen-Image-Edit**](https://huggingface.co/Qwen/Qwen-Image-Edit) together with the [4/8-step Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) variants! Models are available on [Hugging Face](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit). Try them out with our [example script](examples/v1/qwen-image-edit.py).
- **[2025-09-04]** 🚀 Official release of **Nunchaku v1.0.0**! Qwen-Image now supports **asynchronous offloading**, reducing VRAM usage to as little as **3 GiB** with no performance loss. Check out the [tutorial](https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html) to get started.
- **[2025-08-27]** 🔥 Release **4-bit [4/8-step lightning Qwen-Image](https://huggingface.co/lightx2v/Qwen-Image-Lightning)**! Download on [Hugging Face](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image) or [ModelScope](https://modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image), and try it with our [example script](examples/v1/qwen-image-lightning.py).
- **[2025-08-15]** 🔥 Our **4-bit Qwen-Image** models are now live on [Hugging Face](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image)! Get started with our [example script](examples/v1/qwen-image.py). *ComfyUI, LoRA, and CPU offloading support are coming soon!*
- **[2025-08-15]** 🚀 The **Python backend** is now available! Explore our Pythonic FLUX models [here](nunchaku/models/transformers/transformer_flux_v2.py) and see the modular **4-bit linear layer** [here](nunchaku/models/linear.py).
- **[2025-07-31]** 🚀 **[FLUX.1-Krea-dev](https://www.krea.ai/blog/flux-krea-open-source-release) is now supported!** Check out our new [example script](./examples/flux.1-krea-dev.py) to get started.
- **[2025-07-13]** 🚀 The official [**Nunchaku documentation**](https://nunchaku.tech/docs/nunchaku/) is now live! Explore comprehensive guides and resources to help you get started.
- **[2025-06-29]** 🔥 Support **FLUX.1-Kontext**! Try out our [example script](./examples/flux.1-kontext-dev.py) to see it in action! Our demo is available at this [link](https://nunchux.ai/)!
- **[2025-06-01]** 🚀 **Release v0.3.0!** This update adds support for multiple-batch inference, [**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0), initial integration of [**PuLID**](https://github.com/ToTheBeginning/PuLID), and introduces [**Double FB Cache**](examples/flux.1-dev-double_cache.py). You can now load Nunchaku FLUX models as a single file, and our upgraded [**4-bit T5 encoder**](https://huggingface.co/nunchaku-ai/nunchaku-t5) now matches **FP8 T5** in quality!
- **[2025-04-16]** 🎥 Released tutorial videos in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee) to assist installation and usage.
- **[2025-04-09]** 📢 Published the [April roadmap](https://github.com/nunchux-ai/nunchaku/issues/266) and an [FAQ](https://github.com/nunchux-ai/nunchaku/discussions/262) to help the community get started and stay up to date with Nunchaku’s development.
- **[2025-04-05]** 🚀 **Nunchaku v0.2.0 released!** This release brings [**multi-LoRA**](examples/flux.1-dev-multiple-lora.py) and [**ControlNet**](examples/flux.1-dev-controlnet-union-pro.py) support with even faster performance powered by [**FP16 attention**](#fp16-attention) and [**First-Block Cache**](#first-block-cache). We've also added compatibility for [**20-series GPUs**](examples/flux.1-dev-turing.py) — Nunchaku is now more accessible than ever!
- **[2025-03-07]** 🚀 **Nunchaku v0.1.4 Released!** We've supported [4-bit text encoder and per-layer CPU offloading](#Low-Memory-Inference), reducing FLUX's minimum memory requirement to just **4 GiB** while maintaining a **2–3× speedup**. This update also fixes various issues related to resolution, LoRA, pin memory, and runtime stability. Check out the release notes for full details!
- **[2025-02-20]** 🚀 **Support NVFP4 precision on NVIDIA RTX 5090!** NVFP4 delivers superior image quality compared to INT4, offering **~3× speedup** on the RTX 5090 over BF16. Learn more in our [blog](https://hanlab.mit.edu/blog/svdquant-nvfp4), checkout [`examples`](./examples) for usage and try [our demo](https://nunchux.ai/) online!
- **[2025-02-18]** 🔥 [**Customized LoRA conversion**](#Customized-LoRA) and [**model quantization**](#Customized-Model-Quantization) instructions are now available! **[ComfyUI](./comfyui)** workflows now support **customized LoRA**, along with **FLUX.1-Tools**!
- **[2025-02-11]** 🎉 **[SVDQuant](http://arxiv.org/abs/2411.05007) has been selected as a ICLR 2025 Spotlight! FLUX.1-tools Gradio demos are now available!** Check [here](#gradio-demos) for the usage details! Our new [depth-to-image demo](https://nunchux.ai/) is also online—try it out!
- **[2025-02-04]** **🚀 4-bit [FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/) is here!** Enjoy a **2-3× speedup** over the original models. Check out the [examples](./examples) for usage. **ComfyUI integration is coming soon!**
- **[2025-01-23]** 🚀 **4-bit [SANA](https://nvlabs.github.io/Sana/) support is here!** Experience a 2-3× speedup compared to the 16-bit model. Check out the [usage example](examples/sana1.6b_pag.py) and the [deployment guide](app/sana/t2i) for more details. Explore our live demo at [nunchux.ai](https://nunchux.ai/)!
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) has been accepted to **ICLR 2025**!
- **[2024-12-08]** Support [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Please check [ComfyUI-nunchaku](https://github.com/nunchux-ai/ComfyUI-nunchaku) for the usage.
- **[2024-11-07]** 🔥 Our latest **W4A4** Diffusion model quantization work [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) is publicly released! Check [**DeepCompressor**](https://github.com/nunchux-ai/deepcompressor) for the quantization library.

</details>

## Overview

![teaser](https://huggingface.co/datasets/nunchaku-ai/cdn/resolve/main/nunchaku/assets/teaser.jpg)
**Nunchaku** is a high-performance inference engine for low-bit neural networks. It implements **SVDQuant**, a post-training quantization technique for 4-bit weights and activations that well maintains visual fidelity. On 12B FLUX.1-dev, it achieves 3.6× memory reduction compared to the BF16 model. By eliminating CPU offloading, it offers 8.7× speedup over the 16-bit model when on a 16GB laptop 4090 GPU, 3× faster than the NF4 W4A16 baseline. On PixArt-∑, it demonstrates significantly superior visual quality over other W4A4 or even W4A8 baselines. "E2E" means the end-to-end latency including the text encoder and VAE decoder.

**SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**<br>
[Muyang Li](https://lmxyy.me)\*, [Yujun Lin](https://yujunlin.com)\*, [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*, [Tianle Cai](https://www.tianle.website/#/), [Xiuyu Li](https://xiuyuli.com), [Junxian Guo](https://github.com/JerryGJX), [Enze Xie](https://xieenze.github.io), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), and [Song Han](https://hanlab.mit.edu/songhan) <br>
*MIT, NVIDIA, CMU, Princeton, UC Berkeley, SJTU, and Pika Labs* <br>

https://github.com/user-attachments/assets/fdd4ab68-6489-4c65-8768-259bd866e8f8

## Method

#### Quantization Method -- SVDQuant

![intuition](https://huggingface.co/datasets/nunchaku-ai/cdn/resolve/main/nunchaku/assets/intuition.gif)Overview of SVDQuant. Stage1: Originally, both the activation $\boldsymbol{X}$ and weights $\boldsymbol{W}$ contain outliers, making 4-bit quantization challenging. Stage 2: We migrate the outliers from activations to weights, resulting in the updated activation $\hat{\boldsymbol{X}}$ and weights $\hat{\boldsymbol{W}}$. While $\hat{\boldsymbol{X}}$ becomes easier to quantize, $\hat{\boldsymbol{W}}$ now becomes more difficult. Stage 3: SVDQuant further decomposes $\hat{\boldsymbol{W}}$ into a low-rank component $\boldsymbol{L}_1\boldsymbol{L}_2$ and a residual $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$ with SVD. Thus, the quantization difficulty is alleviated by the low-rank branch, which runs at 16-bit precision.

#### Nunchaku Engine Design

![engine](https://huggingface.co/datasets/nunchaku-ai/cdn/resolve/main/nunchaku/assets/engine.jpg) (a) Naïvely running low-rank branch with rank 32 will introduce 57% latency overhead due to extra read of 16-bit inputs in *Down Projection* and extra write of 16-bit outputs in *Up Projection*. Nunchaku optimizes this overhead with kernel fusion. (b) *Down Projection* and *Quantize* kernels use the same input, while *Up Projection* and *4-Bit Compute* kernels share the same output. To reduce data movement overhead, we fuse the first two and the latter two kernels together.

## Performance

![efficiency](https://huggingface.co/datasets/nunchaku-ai/cdn/resolve/main/nunchaku/assets/efficiency.jpg)SVDQuant reduces the 12B FLUX.1 model size by 3.6× and cuts the 16-bit model's memory usage by 3.5×. With Nunchaku, our INT4 model runs 3.0× faster than the NF4 W4A16 baseline on both desktop and laptop NVIDIA RTX 4090 GPUs. Notably, on the laptop 4090, it achieves a total 10.1× speedup by eliminating CPU offloading. Our NVFP4 model is also 3.1× faster than both BF16 and NF4 on the RTX 5090 GPU.

## Getting Started

- [Installation Guide](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
- [Usage Tutorial](https://nunchaku.tech/docs/nunchaku/usage/basic_usage.html)
- [ComfyUI Plugin: ComfyUI-nunchaku](https://github.com/nunchux-ai/ComfyUI-nunchaku)
- [Custom Model Quantization: DeepCompressor](https://github.com/nunchux-ai/deepcompressor)
- [Gradio Demo Apps](https://github.com/nunchux-ai/nunchaku/tree/main/app)
- [Reproduce SVDQuant Paper Results](app/flux.1/t2i)
- [API Reference](https://nunchaku.tech/docs/nunchaku/python_api/nunchaku.html)
- [Contribution Guide](https://nunchaku.tech/docs/nunchaku/developer/contribution_guide.html)
- [Frequently Asked Questions](https://nunchaku.tech/docs/nunchaku/faq/faq.html)

## Contact Us

For enterprises interested in adopting SVDQuant or Nunchaku, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at muyangli@nunchaku.tech.

## Related Projects

- [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
- [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
- [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), MLSys 2025
- [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025
- [Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity](https://arxiv.org/abs/2502.01776), ICML 2025
- [Radial Attention: $O(n \log n)$ Sparse Attention with Energy Decay for Long Video Generation](https://github.com/mit-han-lab/radial-attention), NeurIPS 2025
- [Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation](https://arxiv.org/abs/2505.18875), NeurIPS 2025

## Citation

If you find `nunchaku` useful or relevant to your research, please cite our paper:

```bibtex
@inproceedings{
  li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgments

We thank MIT-IBM Watson AI Lab, MIT and Amazon Science Hub, MIT AI Hardware Program, National Science Foundation, Packard Foundation, Dell, LG, Hyundai, and Samsung for supporting this research. We thank NVIDIA for donating the DGX server. We thank [Yotta Labs](https://www.yottalabs.ai/) for generously sponsoring our computing resources.

We use [img2img-turbo](https://github.com/GaParmar/img2img-turbo) to train the sketch-to-image LoRA. Our text-to-image and image-to-image UI is built upon [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) and [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py), respectively. Our safety checker is borrowed from [hart](https://github.com/mit-han-lab/hart).

Nunchaku is also inspired by many open-source libraries, including (but not limited to) [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [QServe](https://github.com/mit-han-lab/qserve), [AWQ](https://github.com/mit-han-lab/llm-awq), [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), and [Atom](https://github.com/efeslab/Atom).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nunchux-ai/nunchaku&type=Date)](https://www.star-history.com/#nunchux-ai/nunchaku&Date)

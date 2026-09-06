Qwen-Image
==========

Original Qwen-Image
-------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/qwen-image.jpg
   :alt: Qwen-Image with Nunchaku

`Qwen-Image <hf_qwen-image_>`_ is an image generation foundation model in the Qwen series that achieves significant advances in complex text rendering.
Below is a minimal example for running the 4-bit quantized `Qwen-Image <hf_qwen-image_>`_ model with Nunchaku.
Nunchaku offers an API compatible with `Diffusers <github_diffusers_>`_, allowing for a familiar user experience.

.. literalinclude:: ../../../examples/v1/qwen-image.py
    :language: python
    :caption: Running Qwen-Image (`examples/v1/qwen-image.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image.py>`__)
    :linenos:

When using Nunchaku, replace the standard ``QwenImageTransformer2dModel`` with :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel`.
The :meth:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel.from_pretrained` method loads quantized models from either Hugging Face or local file paths.

.. note::

   - The :func:`~nunchaku.utils.get_precision` function automatically detects whether your GPU supports INT4 or FP4 quantization.
     Use FP4 models for Blackwell GPUs (RTX 50-series) and INT4 models for other architectures.
   - Increasing the rank (e.g., to 128) can improve output quality.
   - To reduce VRAM usage, enable asynchronous CPU offloading with :meth:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel.set_offload`. For further savings, you may also enable Diffusers' ``pipeline.enable_sequential_cpu_offload()``, but be sure to exclude ``transformer`` from offloading, as Nunchaku's offloading mechanism differs from Diffusers'. With these settings, VRAM usage can be reduced to approximately 3GB.

Distilled Qwen-Image (Qwen-Image-Lightning)
-------------------------------------------

For faster inference, we provide pre-quantized 4-step and 8-step Qwen-Image models by integrating `Qwen-Image-Lightning LoRAs <hf_qwen-image-lightning_>`_.
See the example script below:

.. literalinclude:: ../../../examples/v1/qwen-image-lightning.py
    :language: python
    :caption: Running Qwen-Image-Lightning (`examples/v1/qwen-image-lightning.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image-lightning.py>`__)
    :linenos:

Custom LoRA support is under development.

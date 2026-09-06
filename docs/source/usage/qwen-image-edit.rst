Qwen-Image-Edit
===============

Original Qwen-Image-Edit
------------------------

`Qwen-Image-Edit <hf_qwen-image-edit_>`_ is the image editing version of Qwen-Image.
Below is a minimal example for running the 4-bit quantized `Qwen-Image-Edit <hf_qwen-image-edit_>`_ model with Nunchaku.
Nunchaku offers an API compatible with `Diffusers <github_diffusers_>`_, allowing for a familiar user experience.

.. literalinclude:: ../../../examples/v1/qwen-image-edit.py
    :language: python
    :caption: Running Qwen-Image-Edit (`examples/v1/qwen-image-edit.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image-edit.py>`__)
    :linenos:

When using Nunchaku, replace the standard ``QwenImageTransformer2dModel`` with :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel`.
The :meth:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel.from_pretrained` method loads quantized models from either Hugging Face or local file paths.

.. note::

   - The :func:`~nunchaku.utils.get_precision` function automatically detects whether your GPU supports INT4 or FP4 quantization.
     Use FP4 models for Blackwell GPUs (RTX 50-series) and INT4 models for other architectures.
   - Increasing the rank (e.g., to 128) can improve output quality.
   - To reduce VRAM usage, enable asynchronous CPU offloading with :meth:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel.set_offload`. For further savings, you may also enable Diffusers' ``pipeline.enable_sequential_cpu_offload()``, but be sure to exclude ``transformer`` from offloading, as Nunchaku's offloading mechanism differs from Diffusers'. With these settings, VRAM usage can be reduced to approximately 3GB.

Distilled Qwen-Image-Edit (Qwen-Image-Lightning)
------------------------------------------------

For faster inference, we provide pre-quantized 4-step and 8-step Qwen-Image-Edit models by integrating `Qwen-Image-Lightning LoRAs <hf_qwen-image-lightning_>`_.
See the example script below:

.. literalinclude:: ../../../examples/v1/qwen-image-edit-lightning.py
    :language: python
    :caption: Running Qwen-Image-Edit-Lightning (`examples/v1/qwen-image-edit-lightning.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image-edit-lightning.py>`__)
    :linenos:

Qwen-Image-Edit-2509
--------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-qwen-image-edit-2509.png
   :alt: Nunchaku-Qwen-Image-Edit-2509

Qwen-Image-Edit-2509 is an monthly iteration of Qwen-Image-Edit.
Below is a minimal example for running the 4-bit quantized `Qwen-Image-Edit-2509 <hf_qwen-image-edit-2509_>`_ model with Nunchaku.

.. literalinclude:: ../../../examples/v1/qwen-image-edit-2509.py
    :language: python
    :caption: Running Qwen-Image-Edit-2509 (`examples/v1/qwen-image-edit-2509.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image-edit-2509.py>`__)
    :linenos:

.. note::
   This example requires ``diffusers`` version 0.36.0 or higher.

Custom LoRA support is under development.

Distilled Qwen-Image-Edit-2509 (Qwen-Image-Edit-2509-Lightning)
---------------------------------------------------------------

For faster inference of the 2509 branch, we provide pre-quantized Lightning variants by integrating `Qwen-Image-Lightning LoRAs <hf_qwen-image-lightning_>`_.
See the example script below:

.. literalinclude:: ../../../examples/v1/qwen-image-edit-2509-lightning.py
    :language: python
    :caption: Running Qwen-Image-Edit-2509-Lightning (`examples/v1/qwen-image-edit-2509-lightning.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image-edit-2509-lightning.py>`__)
    :linenos:

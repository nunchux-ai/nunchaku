Basic Usage
===========

The following is a minimal script for running 4-bit `FLUX.1 <github_flux_>`_ using Nunchaku.
Nunchaku provides the same API as `Diffusers <github_diffusers_>`_, so you can use it in a familiar way.

.. tabs::

   .. tab:: Default (Ampere, Ada, Blackwell, etc.)

      .. literalinclude:: ../../../examples/flux.1-dev.py
         :language: python
         :caption: Running FLUX.1-dev (`examples/flux.1-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev.py>`__)
         :linenos:

   .. tab:: Turing GPUs (e.g., RTX 20 series)

      .. literalinclude:: ../../../examples/flux.1-dev-turing.py
         :language: python
         :caption: Running FLUX.1-dev on Turing GPUs (`examples/flux.1-dev-turing.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-turing.py>`__)
         :linenos:

The key difference when using Nunchaku is replacing the standard ``FluxTransformer2dModel``
with :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`.
The :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.from_pretrained`
method loads quantized models and accepts either Hugging Face remote file paths or local file paths.

.. note::

   The :func:`~nunchaku.utils.get_precision` function automatically detects whether your GPU supports INT4 or FP4 quantization.
   Use FP4 models for Blackwell GPUs (RTX 50-series) and INT4 models for other architectures.

.. note::

   For **Turing GPUs (e.g., NVIDIA 20-series)**, additional configuration is required:

   - Set ``torch_dtype=torch.float16`` in both the transformer and pipeline initialization
   - Use ``transformer.set_attention_impl("nunchaku-fp16")`` to enable FP16 attention
   - Enable offloading with ``offload=True`` in the transformer and ``pipeline.enable_sequential_cpu_offload()`` if you do not have enough VRAM.

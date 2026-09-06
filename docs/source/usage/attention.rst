FP16 Attention
==============

Nunchaku provides an FP16 attention implementation that delivers up to **1.2×** faster performance on NVIDIA 30-, 40-,
and 50-series GPUs compared to FlashAttention-2, without precision loss.

.. literalinclude:: ../../../examples/flux.1-dev-fp16attn.py
   :language: python
   :caption: Running FLUX.1-dev with FP16 Attention (`examples/flux.1-dev-fp16attn.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-fp16attn.py>`__)
   :linenos:
   :emphasize-lines: 11

The key change from `Basic Usage <./basic_usage>`_ is use ``transformer.set_attention_impl("nunchaku-fp16")`` to enable FP16 attention.
While FlashAttention-2 is the default, FP16 attention offers better performance on modern NVIDIA GPUs.
Switch back with ``transformer.set_attention_impl("flashattn2")``.

For more details, see :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.set_attention_impl`.

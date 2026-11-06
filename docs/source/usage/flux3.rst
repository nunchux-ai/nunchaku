FLUX.3
======

The following is the example of running Nunchaku version of FLUX.3 Image Klein text-to-image pipeline.
It requires a diffusers version that provides ``Flux3Transformer2DModel``.

.. tabs::

   .. tab:: FLUX.3 Image Klein

      .. literalinclude:: ../../../examples/v1/flux.3-image-klein.py
         :language: python
         :caption: Running FLUX.3 Image Klein (`examples/v1/flux.3-image-klein.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.3-image-klein.py>`__)
         :linenos:


To use `FP16 Attention <./attention>`_, call ``transformer.set_attention_impl("nunchaku-fp16")`` after loading the
transformer.

For more details, see :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Transformer2DModel`.

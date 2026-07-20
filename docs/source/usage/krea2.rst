Krea 2
======

The following is the example of running Nunchaku version of Krea 2 text-to-image pipeline.

.. tabs::

   .. tab:: Krea-2-Turbo

      .. literalinclude:: ../../../examples/v1/krea-2-turbo.py
         :language: python
         :caption: Running Krea-2-Turbo (`examples/v1/krea-2-turbo.py <https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/krea-2-turbo.py>`__)
         :linenos:

.. note::

   Krea 2 uses grouped-query attention and always passes an attention mask, because text and
   image share one sequence. PyTorch SDPA will not serve ``enable_gqa=True`` together with a
   mask on the flash backend and falls back to the math backend silently, which costs roughly
   3x. The processor therefore expands the key/value heads explicitly.

For more details, see :class:`~nunchaku.models.transformers.transformer_krea2.NunchakuKrea2Transformer2DModel`.

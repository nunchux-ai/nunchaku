Cache
=====

.. _usage-fbcache:

First-Block Cache
-----------------

Nunchaku supports `First-Block Cache (FB Cache) <fbcache>`_ for faster long-step denoising. Example usage:

.. literalinclude:: ../../../examples/flux.1-dev-cache.py
   :language: python
   :caption: Running FLUX.1-dev with FB Cache (`examples/flux.1-dev-cache.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-cache.py>`__)
   :linenos:
   :emphasize-lines: 15-17

Enable it with :func:`~nunchaku.caching.diffusers_adapters.flux.apply_cache_on_pipe`:

.. code-block:: python

    apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)

Adjust ``residual_diff_threshold`` to trade speed for quality - higher values are faster but lower quality.
Recommended value 0.12 gives 2× speedup for 50-step and 1.4× for 30-step denoising.

.. _usage-cache-dit:

Cache-DiT
---------

Nunchaku supports `Cache-DiT <https://github.com/vipshop/cache-dit>`__ to enable further accelerated inference through various caching strategies, such as DB Cache.

Install the required package:

.. code-block:: bash

    pip install cache-dit

.. tabs::

   .. tab:: FLUX.1

      .. literalinclude:: ../../../examples/v1/flux.1-dev-cache-dit.py
         :language: python
         :caption: FLUX.1-dev with Cache-DiT (`examples/v1/flux.1-dev-cache-dit.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.1-dev-cache-dit.py>`__)
         :linenos:

   .. tab:: Qwen-Image

      .. literalinclude:: ../../../examples/v1/qwen-image-cache-dit.py
         :language: python
         :caption: Qwen-Image with Cache-DiT (`examples/v1/qwen-image-cache-dit.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/qwen-image-cache-dit.py>`__)
         :linenos:

For more information, refer to the `Cache-DiT documentation <https://cache-dit.readthedocs.io/en/latest/>`__.

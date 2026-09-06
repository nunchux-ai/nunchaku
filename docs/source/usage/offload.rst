CPU Offload
===========

Nunchaku provides CPU offload capabilities to significantly reduce GPU memory usage with minimal performance impact.
This feature is fully compatible with `Diffusers <diffusers_repo>`_ offload mechanisms.

.. literalinclude:: ../../../examples/flux.1-dev-offload.py
   :language: python
   :caption: Running FLUX.1-dev with CPU Offload (`examples/flux.1-dev-offload.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-offload.py>`__)
   :linenos:
   :emphasize-lines: 9, 13, 14

The following modifications are required compared to `basic usage <../basic_usage/basic_usage>`_:

**Nunchaku CPU Offload** (line 9):
Enable Nunchaku's built-in CPU offload by setting ``offload=True`` during transformer initialization.
This intelligently offloads inactive model components to CPU memory, reducing GPU memory footprint.

**Diffusers Sequential Offload** (line 14):
Activate Diffusers' sequential CPU offload with ``pipeline.enable_sequential_cpu_offload()``.
This provides automatic device management and additional memory optimization.

.. note::
    When using CPU offload, manual device placement with ``.to('cuda')`` is unnecessary,
    as ``pipeline.enable_sequential_cpu_offload()`` handles all device management automatically.

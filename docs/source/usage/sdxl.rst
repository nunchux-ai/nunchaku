Stable Diffusion XL
===================

The following is the example of running Nunchaku INT4 version of SDXL and SDXL-Turbo text-to-image pipeline.

.. tabs::

   .. tab:: SDXL

      .. literalinclude:: ../../../examples/v1/sdxl.py
         :language: python
         :caption: Running Nunchaku SDXL (`examples/v1/sdxl.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/sdxl.py>`__)
         :linenos:

   .. tab:: SDXL-turbo

      .. literalinclude:: ../../../examples/v1/sdxl-turbo.py
         :language: python
         :caption: Running Nunchaku SDXL-Turbo (`examples/v1/sdxl-turbo.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/sdxl-turbo.py>`__)
         :linenos:


For more details, see :class:`~nunchaku.models.unets.unet_sdxl.NunchakuSDXLUNet2DConditionModel`.

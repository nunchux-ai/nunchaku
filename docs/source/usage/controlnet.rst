ControlNets
===========

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/control.jpg
   :alt: ControlNet Integration with Nunchaku

Nunchaku supports mainly two types of ControlNets for FLUX.1.
The first one is `FLUX.1-tools <blog_flux1-tools_>`_ from Black-Forest-Labs.
The second one is the community-contributed ControlNets, like `ControlNet-Union-Pro <hf_cn-union-pro_>`_.

FLUX.1-tools
------------

FLUX.1-tools Base Models
^^^^^^^^^^^^^^^^^^^^^^^^

Nunchaku provides quantized FLUX.1-tools base models.
The implementation follows the same pattern as described in :doc:`Basic Usage <./basic_usage>`,
utilizing an API interface compatible with `Diffusers <github_diffusers_>`_
where the ``FluxTransformer2dModel`` is replaced with :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`.
The primary modification involves switching to the appropriate ControlNet pipeline.
Refer to the following examples for detailed implementation guidance.

.. tabs::

   .. tab:: FLUX.1-Canny-Dev

      .. literalinclude:: ../../../examples/flux.1-canny-dev.py
         :language: python
         :caption: Running FLUX.1-Canny-Dev (`examples/flux.1-canny-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-canny-dev.py>`__)
         :linenos:

   .. tab:: FLUX.1-Depth-Dev

      .. literalinclude:: ../../../examples/flux.1-depth-dev.py
         :language: python
         :caption: Running FLUX.1-Depth-Dev (`examples/flux.1-depth-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-depth-dev.py>`__)
         :linenos:

   .. tab:: FLUX.1-Fill-Dev

      .. literalinclude:: ../../../examples/flux.1-fill-dev.py
         :language: python
         :caption: Running FLUX.1-Fill-Dev (`examples/flux.1-fill-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-fill-dev.py>`__)
         :linenos:

   .. tab:: FLUX.1-Redux-Dev

      .. literalinclude:: ../../../examples/flux.1-redux-dev.py
         :language: python
         :caption: Running FLUX.1-Redux-Dev (`examples/flux.1-redux-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-redux-dev.py>`__)
         :linenos:

FLUX.1-tools LoRAs
^^^^^^^^^^^^^^^^^^

Nunchaku supports FLUX.1-tools LoRAs for converting quantized FLUX.1-dev models to controllable variants.
Implementation follows the same pattern as :doc:`Customized LoRAs <lora>`,
requiring only the ``FluxControlPipeline`` for the target model.

.. tabs::

   .. tab:: FLUX.1-Canny-Dev

      .. literalinclude:: ../../../examples/flux.1-canny-dev-lora.py
         :language: python
         :caption: Running FLUX.1-Canny-Dev-LoRA (`examples/flux.1-canny-dev-lora.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-canny-dev-lora.py>`__)
         :linenos:

   .. tab:: FLUX.1-Depth-Dev

      .. literalinclude:: ../../../examples/flux.1-depth-dev-lora.py
         :language: python
         :caption: Running FLUX.1-Depth-Dev-LoRA (`examples/flux.1-depth-dev-lora.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-depth-dev-lora.py>`__)
         :linenos:

ControlNet-Union-Pro
--------------------

`ControlNet-Union-Pro <hf_cn-union-pro_>`_ is a community-developed ControlNet implementation for FLUX.1.
Unlike FLUX.1-tools that directly fine-tunes the model to incorporate control signals,
`ControlNet-Union-Pro <hf_cn-union-pro_>`_ uses additional control modules.
It provides native support for multiple control types including Canny edges and depth maps.

Nunchaku currently executes these control modules at their original precision levels.
The following example demonstrates running `ControlNet-Union-Pro <hf_cn-union-pro_>`_ with Nunchaku.

.. literalinclude:: ../../../examples/flux.1-dev-controlnet-union-pro.py
   :language: python
   :caption: Running ControlNet-Union-Pro (`examples/flux.1-dev-controlnet-union-pro.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-controlnet-union-pro.py>`__)
   :linenos:

Usage for `ControlNet-Union-Pro2 <hf_cn-union-pro2_>`_ is similar.
Quantized ControlNet support is currently in development. Stay tuned!

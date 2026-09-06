IP Adapter
==========

.. warning::
  IP Adapter support will be deprecated in March, 2026.

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-ip-adapter.png

Nunchaku supports `IP Adapter <hf_ip-adapterv2_>`_, an adapter achieving image prompt capability for the FLUX.1-dev

.. literalinclude:: ../../../examples/flux.1-dev-IP-adapter.py
   :language: python
   :caption: IP Adapter Example (`examples/flux.1-dev-IP-adapter.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-IP-adapter.py>`__)
   :linenos:

The IP Adapter integration in Nunchaku follows these main steps:

**Model Initialization**:

- Load a Nunchaku FLUX.1-dev transformer model using :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.from_pretrained`.
- Initialize the FLUX pipeline with :class:`diffusers.FluxPipeline`, passing the transformer and setting the appropriate precision.

**IP Adapter Loading**:

- Use ``pipeline.load_ip_adapter`` to load the IP Adapter weights and the CLIP image encoder.

  - ``pretrained_model_name_or_path_or_dict``: Hugging Face repo or local path for the IP Adapter weights.
  - ``weight_name``: Name of the weights file (e.g., ``ip_adapter.safetensors``).
  - ``image_encoder_pretrained_model_name_or_path``: Name or path of the CLIP image encoder.
  - Apply the IP Adapter to the pipeline with :func:`~nunchaku.models.ip_adapter.diffusers_adapters.apply_IPA_on_pipe`, specifying the adapter scale and repo ID.

**Caching (Optional)**:

Enable caching for faster inference and reduced memory usage with :func:`~nunchaku.caching.diffusers_adapters.apply_cache_on_pipe`. See :doc:`cache` for more details.

**Image Generation**:

- Load the image to be used as the image prompt (IP Adapter reference).
- Call the pipeline with:

  - ``prompt``: The text prompt for generation.
  - ``ip_adapter_image``: The reference image (must be RGB).
- The output image will reflect both the text prompt and the visual style/content of the reference image.

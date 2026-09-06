PuLID
=====

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-pulid.png

Nunchaku integrates `PuLID <paper_pulid_>`_, a tuning-free identity customization method for text-to-image generation.
This feature allows you to generate images that maintain specific identity characteristics from reference photos.

.. literalinclude:: ../../../examples/flux.1-dev-pulid.py
   :language: python
   :caption: PuLID Example (`examples/flux.1-dev-pulid.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-pulid.py>`__)
   :linenos:

The PuLID integration follows these key steps:

**Model Initialization** (lines 12-20):
Load a Nunchaku FLUX.1-dev model using :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.from_pretrained`
and initialize the FLUX PuLID pipeline with :class:`~nunchaku.pipeline.pipeline_flux_pulid.PuLIDFluxPipeline`.

**Forward Method Override** (line 22):
Replace the transformer's forward method with PuLID's specialized implementation using
``MethodType(pulid_forward, pipeline.transformer)``.
This modification enables identity-aware generation capabilities.
See :meth:`~nunchaku.models.pulid.pulid_forward.pulid_forward` for more details.

**Reference Image Processing** (line 24):
Load and prepare the reference identity image that will guide the generation process.
This image defines the identity characteristics to be preserved in the output.

**Identity-Controlled Generation** (lines 26-32):
Execute the pipeline with identity-specific parameters:

- ``id_image``: The reference identity image
- ``id_weight``: Identity influence strength (range: 0.0-1.0, where 1.0 provides maximum identity preservation)
- Standard generation parameters (prompt, inference steps, guidance scale)

The generated image will incorporate the identity features from the reference photo while adhering to the provided text prompt.

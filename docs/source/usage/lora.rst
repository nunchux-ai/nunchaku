Customized LoRAs
================

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/lora.jpg
   :alt: LoRA integration with Nunchaku

Single LoRA
-----------

`Nunchaku <github_nunchaku_>`_ seamlessly integrates with off-the-shelf LoRAs without requiring requantization.
Instead of fusing the LoRA branch into the main branch, we directly concatenate the LoRA weights to our low-rank branch.
As Nunchaku uses fused kernel, the overhead of a separate low-rank branch is largely reduced.
Below is an example of running FLUX.1-dev with `Ghibsky <hf_lora_ghibsky_>`_ LoRA.

.. literalinclude:: ../../../examples/flux.1-dev-lora.py
   :language: python
   :caption: Running FLUX.1-dev with `Ghibsky <hf_lora_ghibsky_>`_  LoRA (`examples/flux.1-dev-lora.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-lora.py>`__)
   :linenos:
   :emphasize-lines: 16-19

The LoRA integration in Nunchaku works through two key methods:

**Loading LoRA Parameters** (lines 16-17):
The :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.update_lora_params` method loads LoRA weights from a safetensors file. It supports:

- **Local file path**: ``"/path/to/your/lora.safetensors"``
- **HuggingFace repository with specific file**: ``"aleksa-codes/flux-ghibsky-illustration/lora.safetensors"``.
  The system automatically downloads and caches the LoRA file on first access.

**Controlling LoRA Strength** (lines 18-19):
The :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.set_lora_strength` method sets the LoRA strength parameter,
which controls how much influence the LoRA has on the final output.
A value of 1.0 applies the full LoRA effect, while lower values (e.g., 0.5) apply a more subtle influence.

Multiple LoRAs
--------------

To load multiple LoRAs simultaneously, Nunchaku provides the :func:`~nunchaku.lora.flux.compose.compose_lora` function,
which combines multiple LoRA weights into a single composed LoRA before loading.
This approach enables efficient multi-LoRA inference without requiring separate loading operations.

The following example demonstrates how to compose and load multiple LoRAs:

.. literalinclude:: ../../../examples/flux.1-dev-multiple-lora.py
   :language: python
   :caption: Running FLUX.1-dev with `Ghibsky <hf_lora_ghibsky_>`_ and `FLUX-Turbo <hf_lora_flux-turbo_>`_ LoRA (`examples/flux.1-dev-multiple-lora.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-multiple-lora.py>`__)
   :linenos:
   :emphasize-lines: 17-23

The :func:`~nunchaku.lora.flux.compose.compose_lora` function accepts a list of tuples, where each tuple contains:

- **LoRA path**: Either a local file path or HuggingFace repository path with specific file
- **Strength value**: A float value (typically between 0.0 and 1.0) that controls the influence of that specific LoRA

This composition method allows for precise control over individual LoRA strengths while maintaining computational efficiency through a single loading operation.

.. warning::

   When using multiple LoRAs,
   the :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.set_lora_strength` method
   applies a uniform strength value across all loaded LoRAs, which may not provide the desired level of control.
   For precise management of individual LoRA influences, specify strength values for each LoRA within the
   :func:`~nunchaku.lora.flux.compose.compose_lora` function call.

.. warning::

   Nunchaku's current implementation maintains the LoRA branch separately from the main branch.
   This design choice may impact inference performance when the composed rank becomes large (e.g., > 256).
   A future release will include quantization tools to fuse the LoRA branch into the main branch.

LoRA Conversion
---------------

Nunchaku utilizes the `Diffusers <github_diffusers_>`_ LoRA format as an intermediate representation for converting LoRAs to Nunchaku's native format.
Both the :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.update_lora_params` method and :func:`~nunchaku.lora.flux.compose.compose_lora`
function internally invoke :func:`~nunchaku.lora.flux.diffusers_converter.to_diffusers` to convert LoRAs to the `Diffusers <github_diffusers_>`_ format.
If LoRA functionality is not working as expected, verify that the LoRA has been properly converted to the `Diffusers <github_diffusers_>`_ format.
Please check :func:`~nunchaku.lora.flux.diffusers_converter.to_diffusers` for more details.

Following the conversion to `Diffusers <github_diffusers_>`_ format,
the :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.update_lora_params`
method calls the :func:`~nunchaku.lora.flux.nunchaku_converter.to_nunchaku` function
to perform the final conversion to Nunchaku's format.

Exporting Converted LoRAs
-------------------------

The current implementation employs single-threaded conversion, which may result in extended processing times, particularly for large LoRA files.
To address this limitation, users can pre-compose LoRAs using the :mod:`nunchaku.lora.flux.compose` command-line interface.
The syntax is as follows:

.. code-block:: bash

   python -m nunchaku.lora.flux.compose -i lora1.safetensors lora2.safetensors -s 0.8 0.6 -o composed_lora.safetensors

**Arguments**:

- ``-i``, ``--input-paths``: Paths to the LoRA safetensors files (supports multiple files)
- ``-s``, ``--strengths``: Strength values for each LoRA (must correspond to the number of input files)
- ``-o``, ``--output-path``: Output path for the composed LoRA safetensors file

This command composes the specified LoRAs with their respective strength values and saves the result to the output file,
which can subsequently be loaded using :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.update_lora_params` for optimized inference performance.

Following composition, users may either load the file directly
(via the ComfyUI LoRA loader or :meth:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.update_lora_params`)
or utilize :mod:`nunchaku.lora.flux.convert` to convert the composed LoRA to Nunchaku's format and export it.
The syntax is as follows:

.. code-block:: bash

   python -m nunchaku.lora.flux.convert --lora-path composed_lora.safetensors --quant-path mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors --output-root ./converted --dtype bfloat16

**Arguments**:

- ``--lora-path``: Path to the LoRA weights safetensor file (required)
- ``--quant-path``: Path to the quantized model safetensor file (default: ``mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors``)
- ``--output-root``: Root directory for the output safetensor file (default: parent directory of the lora file)
- ``--lora-name``: Name of the LoRA weights (optional, auto-generated if not provided)
- ``--dtype``: Data type of the converted weights, either ``bfloat16`` or ``float16`` (default: ``bfloat16``)

This command converts the LoRA to Nunchaku's format and saves it with an appropriate filename based on the quantization precision (fp4 or int4).


.. warning::

   LoRAs in Nunchaku format should not be composed with other LoRAs. Additionally, LoRA strength values are permanently embedded in the composed LoRA. To apply different strength values, the LoRAs must be recomposed.

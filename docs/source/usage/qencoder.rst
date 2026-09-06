Quantized Text Encoders
=======================

Nunchaku provides a quantized T5 encoder for FLUX.1 to reduce GPU memory usage.

.. literalinclude:: ../../../examples/flux.1-dev-qencoder.py
   :language: python
   :caption: Running FLUX.1-dev with Quantized T5 (`examples/flux.1-dev-qencoder.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/flux.1-dev-qencoder.py>`__)
   :linenos:
   :emphasize-lines: 11, 14

The key changes from `Basic Usage <./basic_usage>`_ are:

**Loading Quantized T5 Encoder** (line 11):
Use :class:`~nunchaku.models.text_encoders.t5_encoder.NunchakuT5EncoderModel` to load the quantized encoder.
This reduces GPU memory usage while maintaining quality. Supports local or Hugging Face remote paths.

**Pipeline Integration** (line 14):
Pass the quantized encoder to the pipeline via the ``text_encoder_2`` parameter,
replacing the default T5 encoder.

.. note::

   The quantized T5 encoder currently only supports CUDA backend. Turing GPUs will be supported later.

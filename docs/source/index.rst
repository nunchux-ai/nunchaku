Nunchaku Documentation
======================
**Nunchaku** is a high-performance inference engine optimized for low-bit diffusion models and LLMs,
as introduced in our paper `SVDQuant <paper_svdquant_>`_.
Check out `DeepCompressor <github_deepcompressor_>`_ for the quantization library.

.. toctree::
   :maxdepth: 2
   :caption: Installation

   installation/installation.rst
   installation/setup_windows.rst

.. toctree::
    :maxdepth: 1
    :caption: Usage Tutorials

    usage/basic_usage.rst
    usage/qwen-image.rst
    usage/qwen-image-edit.rst
    usage/lora.rst
    usage/kontext.rst
    usage/sdxl.rst
    usage/controlnet.rst
    usage/qencoder.rst
    usage/offload.rst
    usage/attention.rst
    usage/cache.rst
    usage/pulid.rst
    usage/ip_adapter.rst
    usage/zimage.rst

.. toctree::
    :maxdepth: 1
    :caption: Python API Reference

    python_api/nunchaku.rst

.. toctree::
    :maxdepth: 1
    :caption: Useful Tools
    :titlesonly:

    ComfyUI Plugin: ComfyUI-nunchaku <https://nunchaku.tech/docs/ComfyUI-nunchaku/>
    Custom Model Quantization: DeepCompressor <https://github.com/nunchux-ai/deepcompressor>
    Gradio Demos <https://github.com/nunchux-ai/nunchaku/tree/main/app>


.. toctree::
    :maxdepth: 1
    :caption: Other Resources

    faq/faq.rst
    developer/contribution_guide.rst
    developer/docstring.rst
    developer/build_docs.rst

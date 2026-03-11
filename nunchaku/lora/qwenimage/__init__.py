"""
Qwen Image LoRA module for Nunchaku.

This module provides utilities for loading, converting, and composing LoRA weights
for Qwen Image models in Nunchaku's quantized inference framework.

Main Functions
--------------
- :func:`to_diffusers` : Convert LoRA to Diffusers format
- :func:`to_nunchaku` : Convert LoRA to Nunchaku format
- :func:`compose_lora` : Compose multiple LoRAs
- :func:`is_nunchaku_format` : Check if LoRA is in Nunchaku format
- :func:`convert_to_nunchaku_qwenimage_lowrank_dict` : Convert full model LoRA

Example Usage
-------------
>>> from nunchaku.lora.qwenimage import to_nunchaku, compose_lora
>>> # Convert a single LoRA
>>> lora_nunchaku = to_nunchaku("lora.safetensors", output_path="lora_nunchaku.safetensors")
>>> # Compose multiple LoRAs
>>> composed = compose_lora([("lora1.safetensors", 0.8), ("lora2.safetensors", 0.6)])
>>> # Convert the composed LoRA to Nunchaku format
>>> lora_nunchaku = to_nunchaku(composed, output_path="composed_nunchaku.safetensors")
"""

from .compose import compose_lora
from .diffusers_converter import to_diffusers
from .nunchaku_converter import convert_to_nunchaku_qwenimage_lowrank_dict, fuse_vectors, to_nunchaku
from .utils import is_nunchaku_format

__all__ = [
    "to_diffusers",
    "to_nunchaku",
    "compose_lora",
    "convert_to_nunchaku_qwenimage_lowrank_dict",
    "fuse_vectors",
    "is_nunchaku_format",
]

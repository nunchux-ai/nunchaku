"""
This module provides the Nunchaku version of the FLUX.3 transformer (``Flux3Transformer2DModel`` in diffusers).
"""

import itertools
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from diffusers import Flux3Transformer2DModel
from diffusers.models.transformers.transformer_flux3 import Flux3Attention
from huggingface_hub import utils

from ...utils import check_hardware_compatibility, get_precision_from_quantization_config
from ..attention import NunchakuBaseAttention, _patch_linear
from ..attention_processors.flux3 import NunchakuFlux3FA2Processor, NunchakuFlux3FP16AttnProcessor
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key


class NunchakuFlux3Attention(NunchakuBaseAttention):
    """
    Nunchaku-optimized Flux3Attention module with a quantized and fused QKV projection.

    Parameters
    ----------
    other : Flux3Attention
        The original Flux3Attention module to wrap and quantize.
    processor : str, optional
        The attention processor to use ("flashattn2" or "nunchaku-fp16").
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: Flux3Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuFlux3Attention, self).__init__(processor)
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.heads = other.heads
        self.dropout = other.dropout
        self.use_bias = other.use_bias

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k

        # Fuse the QKV projections for efficiency.
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        self.to_out = other.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for NunchakuFlux3Attention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        attention_mask : torch.Tensor, optional
            Attention mask.
        image_rotary_emb : tuple of torch.Tensor, optional
            ``(cos, sin)`` rotary embeddings of the tokens.
        **kwargs
            Additional arguments.

        Returns
        -------
        Output of the attention processor.
        """
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor ("flashattn2" or "nunchaku-fp16").

            - ``"flashattn2"``: Attention through diffusers' ``dispatch_attention_fn`` with its default backend (PyTorch scaled dot-product attention, which runs FlashAttention-2 kernels for 16-bit inputs). See :class:`~nunchaku.models.attention_processors.flux3.NunchakuFlux3FA2Processor`.
            - ``"nunchaku-fp16"``: Uses FP16 attention accumulation, faster than FlashAttention-2 on NVIDIA 30-, 40-, and 50-series GPUs. See :class:`~nunchaku.models.attention_processors.flux3.NunchakuFlux3FP16AttnProcessor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuFlux3FA2Processor()
        elif processor == "nunchaku-fp16":
            self.processor = NunchakuFlux3FP16AttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuFlux3Transformer2DModel(Flux3Transformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized Flux3Transformer2DModel.

    In every per-mode and joint transformer block, the attention and feed-forward projections are
    :class:`~nunchaku.models.linear.SVDQW4A4Linear` layers. The attention is a
    :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Attention`, whose QKV projection, QK
    normalization and RoPE run in one fused kernel.
    """

    # The fused QKV kernel reads the QK normalization weights in the activation dtype.
    _keep_in_fp32_modules = None

    def _patch_model(self, **kwargs):
        """
        Replace the attention of every transformer block with
        :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Attention` and its feed-forward
        projections with quantized linears.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization (e.g., precision, rank).

        Returns
        -------
        self : NunchakuFlux3Transformer2DModel
            The patched model.
        """
        for block in itertools.chain(self.txt_mode_blocks, *self.mode_blocks.values(), self.single_transformer_blocks):
            block.attn = NunchakuFlux3Attention(block.attn, **kwargs)
            _patch_linear(block.ff, SVDQW4A4Linear, **kwargs)
        return self

    def set_attention_impl(self, impl: str):
        """
        Set the attention implementation of every transformer block.

        Parameters
        ----------
        impl : str
            Attention implementation to use ("flashattn2" or "nunchaku-fp16").
            See :meth:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Attention.set_processor`.
        """
        for module in self.modules():
            if isinstance(module, NunchakuFlux3Attention):
                module.set_processor(impl)

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuFlux3Transformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuFlux3Transformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        ValueError
            If the checkpoint's quantization does not match the GPU (e.g., INT4 on an RTX 50-series GPU).
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for Flux3Transformer2DModel")

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata["quantization_config"])
        check_hardware_compatibility(quantization_config, device)
        precision = get_precision_from_quantization_config(quantization_config)
        rank = quantization_config.get("rank", 32)
        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        transformer.load_state_dict(model_state_dict)

        return transformer

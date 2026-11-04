"""
This module provides the Nunchaku version of the FLUX.3 transformer (``Flux3Transformer2DModel`` in diffusers).
"""

import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import Flux3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import (
    Flux3Attention,
    Flux3FeedForward,
    Flux3ModeTransformerBlock,
    Flux3SingleTransformerBlock,
)
from huggingface_hub import utils
from torch import nn

from ...ops.misc import layer_norm, layer_norm_, mul_add_, silu_and_mul
from ...utils import check_hardware_compatibility, get_precision_from_quantization_config
from ..attention import NunchakuBaseAttention
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


class NunchakuFlux3FeedForward(nn.Module):
    """
    Quantized Flux3FeedForward: :class:`~nunchaku.models.linear.SVDQW4A4Linear` projections around a SwiGLU
    activation computed by :func:`~nunchaku.ops.misc.silu_and_mul`.

    Parameters
    ----------
    ff : Flux3FeedForward
        The original feed-forward block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, ff: Flux3FeedForward, **kwargs):
        super().__init__()
        self.linear_in = SVDQW4A4Linear.from_linear(ff.linear_in, **kwargs)
        self.linear_out = SVDQW4A4Linear.from_linear(ff.linear_out, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(silu_and_mul(self.linear_in(x)))


def _modulate_(x: torch.Tensor, temb_mod: torch.Tensor) -> torch.Tensor:
    """In-place ``x = x * (1 + scale) + shift`` with ``(shift, scale, gate) = temb_mod.chunk(3)``."""
    shift, scale, _ = torch.chunk(temb_mod, 3, dim=-1)
    return mul_add_(x, scale, shift, 1.0)


def _gated_residual_(x: torch.Tensor, temb_mod: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """In-place ``x = residual + gate * x`` with ``(shift, scale, gate) = temb_mod.chunk(3)``."""
    _, _, gate = torch.chunk(temb_mod, 3, dim=-1)
    return mul_add_(x, gate, residual)


class NunchakuFlux3ModeTransformerBlock(Flux3ModeTransformerBlock):
    """
    Nunchaku-optimized Flux3ModeTransformerBlock with quantized attention and feed-forward layers.

    The layer normalization, the modulation and the gated residual run in Nunchaku's CUDA kernels
    (:mod:`nunchaku.ops.misc`). Unlike the diffusers block, ``temb_mod`` is one ``(B, 1, 3 * D)`` vector per sample, as
    computed by :meth:`NunchakuFlux3Transformer2DModel.forward`.

    Parameters
    ----------
    block : Flux3ModeTransformerBlock
        The original block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: Flux3ModeTransformerBlock, **kwargs):
        super(Flux3ModeTransformerBlock, self).__init__()
        self.norm = block.norm
        self.attn = NunchakuFlux3Attention(block.attn, **kwargs)
        self.ff = NunchakuFlux3FeedForward(block.ff, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for the per-mode transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, S, D)
            Input hidden states.
        temb_mod : torch.Tensor, shape (B, 1, 3 * D)
            Shift, scale and gate of the modulation, shared by the tokens.
        image_rotary_emb : tuple of torch.Tensor
            ``(cos, sin)`` rotary embeddings of the tokens.

        Returns
        -------
        torch.Tensor, shape (B, S, D)
            Output hidden states.
        """
        hidden_states = hidden_states.contiguous()
        norm_hidden_states = _modulate_(layer_norm(hidden_states, self.norm.eps), temb_mod)
        output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)
        output.add_(self.ff(norm_hidden_states))
        return _gated_residual_(output, temb_mod, hidden_states)


class NunchakuFlux3SingleTransformerBlock(Flux3SingleTransformerBlock):
    """
    Nunchaku-optimized Flux3SingleTransformerBlock with quantized attention and feed-forward layers.

    The layer normalization, the modulation and the gated residual run in Nunchaku's CUDA kernels
    (:mod:`nunchaku.ops.misc`). Unlike the diffusers block, ``temb_mod`` is a list of ``(num_tokens, (B, 1, 3 * D))``
    content segments, as computed by :meth:`NunchakuFlux3Transformer2DModel.forward`.

    Parameters
    ----------
    block : Flux3SingleTransformerBlock
        The original block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: Flux3SingleTransformerBlock, **kwargs):
        super(Flux3SingleTransformerBlock, self).__init__()
        self.norm = block.norm
        self.attn = NunchakuFlux3Attention(block.attn, **kwargs)
        self.ff = NunchakuFlux3FeedForward(block.ff, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod: list[Tuple[int, torch.Tensor]],
        temb_mod_context: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the joint transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, S, D)
            Content tokens (image tokens, then reference-image tokens).
        encoder_hidden_states : torch.Tensor, shape (B, T, D)
            Text tokens.
        temb_mod : list of (int, torch.Tensor)
            Modulation of consecutive content segments: ``(num_tokens, (B, 1, 3 * D) shift/scale/gate)``.
        temb_mod_context : torch.Tensor, shape (B, 1, 3 * D)
            Modulation of the text tokens.
        image_rotary_emb : tuple of torch.Tensor
            ``(cos, sin)`` rotary embeddings of the text and content tokens.

        Returns
        -------
        tuple of torch.Tensor
            ``(encoder_hidden_states, hidden_states)``.
        """
        num_txt_tokens = encoder_hidden_states.shape[1]
        combined = layer_norm_(torch.cat([encoder_hidden_states, hidden_states], dim=1), self.norm.eps)
        _modulate_(combined[:, :num_txt_tokens], temb_mod_context)
        start = num_txt_tokens
        for num_tokens, mod in temb_mod:
            _modulate_(combined[:, start : start + num_tokens], mod)
            start += num_tokens

        output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb)
        output.add_(self.ff(combined))

        _gated_residual_(output[:, :num_txt_tokens], temb_mod_context, encoder_hidden_states)
        start = 0
        for num_tokens, mod in temb_mod:
            _gated_residual_(
                output[:, num_txt_tokens + start : num_txt_tokens + start + num_tokens],
                mod,
                hidden_states[:, start : start + num_tokens],
            )
            start += num_tokens
        return output[:, :num_txt_tokens], output[:, num_txt_tokens:]


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
        Replace the per-mode and joint transformer blocks with
        :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3ModeTransformerBlock` and
        :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3SingleTransformerBlock`.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization (e.g., precision, rank).

        Returns
        -------
        self : NunchakuFlux3Transformer2DModel
            The patched model.
        """
        for blocks in itertools.chain([self.txt_mode_blocks], self.mode_blocks.values()):
            for i, block in enumerate(blocks):
                blocks[i] = NunchakuFlux3ModeTransformerBlock(block, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFlux3SingleTransformerBlock(block, **kwargs)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: Optional[torch.Tensor] = None,
        image_cond_ids: Optional[torch.Tensor] = None,
        target_timestep: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for the NunchakuFlux3Transformer2DModel.
        Adapted from diffusers.models.transformers.transformer_flux3.Flux3Transformer2DModel#forward.

        diffusers expands the timestep embedding of each token mode (image, reference image, text) over its tokens
        before the modulation projections. Here every mode's modulation is computed once, as a ``(B, 1, 3 * D)``
        vector, and the blocks broadcast it over the tokens.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, image_seq_len, in_channels)
            Noisy image latent tokens.
        encoder_hidden_states : torch.Tensor, shape (B, text_seq_len, joint_attention_dim)
            Text context embeddings.
        timestep : torch.Tensor, shape (B,)
            Flow timestep in [0, 1] of the image tokens.
        img_ids : torch.Tensor, shape (image_seq_len, 4)
            Position ids of the image tokens.
        txt_ids : torch.Tensor, shape (text_seq_len, 4)
            Position ids of the text tokens.
        image_cond_hidden_states : torch.Tensor, shape (B, cond_seq_len, cond_in_channels), optional
            VAE-encoded reference image tokens.
        image_cond_ids : torch.Tensor, shape (cond_seq_len, 4), optional
            Position ids of the reference image tokens. Required with ``image_cond_hidden_states``.
        target_timestep : torch.Tensor, shape (B,), optional
            Flow timestep the current step jumps to. Required by, and only by, a flow-map model.
        attention_kwargs : dict, optional
            Not supported (LoRA scaling).
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).

        Returns
        -------
        Transformer2DModelOutput or tuple
            The predicted velocity of the image tokens.

        Raises
        ------
        NotImplementedError
            If ``attention_kwargs`` is given, or gradient checkpointing is enabled while gradients are on.
        ValueError
            If the reference image or the target timestep arguments are inconsistent.
        """
        if attention_kwargs:
            raise NotImplementedError("attention_kwargs is not supported for NunchakuFlux3Transformer2DModel")
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            raise NotImplementedError("Gradient checkpointing is not supported for NunchakuFlux3Transformer2DModel")
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError("`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`).")

        batch_size, image_seq_len = hidden_states.shape[:2]
        dtype = hidden_states.dtype

        # 1. Timestep embeddings, one (B, 1, D) vector per token mode: the image tokens at `timestep`, the text
        #    context and the reference tokens at 0
        image_temb = self.time_proj(timestep.to(dtype) * 1000)
        image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb_act = F.silu(image_temb)
        zero_temb_act = F.silu(zero_temb)

        # 2. Text context
        txt_early_mod = self.early_stream_modulations["txt"](zero_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](zero_temb_act)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_rotary_emb = self.pos_embed(txt_ids)

        # 3. Image tokens
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = self.x_embedders["image"](hidden_states)
        image_rotary_emb = self.pos_embed(img_ids)

        # 4. Reference image tokens (optional)
        if image_cond_hidden_states is not None:
            cond_early_mod = self.early_stream_modulations["image_cond"](zero_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](zero_temb_act)
            image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
            cond_rotary_emb = self.pos_embed(image_cond_ids)

        # 5. Per-mode transformer blocks (each mode self-attends within itself only)
        for block in self.txt_mode_blocks:
            encoder_hidden_states = block(encoder_hidden_states, txt_early_mod, txt_rotary_emb)
        for block in self.mode_blocks["image"]:
            hidden_states = block(hidden_states, image_early_mod, image_rotary_emb)
        if image_cond_hidden_states is not None:
            for block in self.mode_blocks["image_cond"]:
                image_cond_hidden_states = block(image_cond_hidden_states, cond_early_mod, cond_rotary_emb)

        # 6. Joint single transformer blocks over [txt, image, image_cond]
        content_mod = [(image_seq_len, image_single_mod)]
        if image_cond_hidden_states is not None:
            content_states = torch.cat([hidden_states, image_cond_hidden_states], dim=1)
            content_mod.append((image_cond_hidden_states.shape[1], cond_single_mod))
            joint_rotary_emb = (
                torch.cat([txt_rotary_emb[0], image_rotary_emb[0], cond_rotary_emb[0]], dim=0),
                torch.cat([txt_rotary_emb[1], image_rotary_emb[1], cond_rotary_emb[1]], dim=0),
            )
        else:
            content_states = hidden_states
            joint_rotary_emb = (
                torch.cat([txt_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([txt_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod=content_mod,
                temb_mod_context=txt_single_mod,
                image_rotary_emb=joint_rotary_emb,
            )

        # 7. Output layers: the image tokens sit first in the content sequence
        hidden_states = content_states[:, :image_seq_len]
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

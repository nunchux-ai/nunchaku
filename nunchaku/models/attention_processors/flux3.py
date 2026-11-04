"""
Attention processor implementations for :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Attention`.
"""

import math
from typing import Optional, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux3 import Flux3AttnProcessor

from ..._C.ops import attention_fp16
from ...ops.fused import fused_qkv_norm_rottary
from ...utils import pad_tensor
from ..embeddings import pack_rotemb


def pack_flux3_rotary_emb(image_rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Pack FLUX.3 rotary embeddings for :func:`~nunchaku.ops.fused.fused_qkv_norm_rottary`.

    The packed tensor is cached on the cosine tensor, so the blocks that share one rotary embedding pack it once. The
    cache is released together with the tensor.

    Parameters
    ----------
    image_rotary_emb : tuple of torch.Tensor
        ``(cos, sin)``, each of shape (S, D), dtype float32, as returned by diffusers' ``Flux3PosEmbed``.

    Returns
    -------
    torch.Tensor, shape (1, S_pad, D), dtype float32
        Packed rotary embedding, with S padded to a multiple of 256.
    """
    cos, sin = image_rotary_emb
    packed = getattr(cos, "_nunchaku_packed_rotemb", None)
    if packed is None:
        # FLUX.3 uses repeat_interleave_real=True, so both entries of a rotary pair share one cos/sin value.
        rotemb = torch.stack([sin[:, 0::2], cos[:, 0::2]], dim=-1).unsqueeze(0).unsqueeze(-2)
        packed = pack_rotemb(pad_tensor(rotemb.float(), 256, 1))
        cos._nunchaku_packed_rotemb = packed
    return packed


class NunchakuFlux3FA2Processor(Flux3AttnProcessor):
    """
    Fused attention processor for FLUX.3.
    Adapted from diffusers.models.transformers.transformer_flux3.Flux3AttnProcessor.

    The quantized QKV projection, the QK RMS normalization and the rotary embedding run in one kernel
    (:func:`~nunchaku.ops.fused.fused_qkv_norm_rottary`); attention runs through diffusers' attention dispatcher.
    """

    # Adapted from diffusers.models.transformers.transformer_flux3.Flux3AttnProcessor#__call__
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for fused attention.

        Parameters
        ----------
        attn : :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Attention`
            Attention module.
        hidden_states : torch.Tensor, shape (B, S, C)
            Input hidden states.
        attention_mask : torch.Tensor, optional
            Attention mask.
        image_rotary_emb : tuple of torch.Tensor
            ``(cos, sin)`` rotary embeddings, each of shape (S, D).

        Returns
        -------
        torch.Tensor, shape (B, S, C)
            Output hidden states.
        """
        qkv = fused_qkv_norm_rottary(
            hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k, pack_flux3_rotary_emb(image_rotary_emb)
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class NunchakuFlux3FP16AttnProcessor:
    """
    Fused attention processor for FLUX.3 with custom Nunchaku FP16 accumulation.
    This is faster than the :class:`~nunchaku.models.attention_processors.flux3.NunchakuFlux3FA2Processor`.

    Parameters
    ----------
    pad_size : int, optional
        Padding size for sequence length. Default is 256.
    """

    def __init__(self, pad_size: int = 256):
        self.pad_size = pad_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Nunchaku FP16 attention.

        Parameters
        ----------
        attn : :class:`~nunchaku.models.transformers.transformer_flux3.NunchakuFlux3Attention`
            Attention module.
        hidden_states : torch.Tensor, shape (B, S, C)
            Input hidden states.
        attention_mask : torch.Tensor, optional
            Not supported. Must be None.
        image_rotary_emb : tuple of torch.Tensor
            ``(cos, sin)`` rotary embeddings, each of shape (S, D).

        Returns
        -------
        torch.Tensor, shape (B, S, C)
            Output hidden states.

        Raises
        ------
        NotImplementedError
            If attention_mask is not None.

        Notes
        -----
        - B: batch size
        - S: sequence length
        - C: channels (C = heads * head_dim)
        """
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")

        pad_size = self.pad_size
        batch_size, num_tokens, channels = hidden_states.shape
        assert channels == attn.heads * attn.head_dim
        num_tokens_pad = math.ceil(num_tokens / pad_size) * pad_size

        query = torch.empty(
            batch_size, attn.heads, num_tokens_pad, attn.head_dim, dtype=torch.float16, device=hidden_states.device
        )
        key = torch.empty_like(query)
        value = torch.empty_like(query)
        fused_qkv_norm_rottary(
            hidden_states,
            attn.to_qkv,
            attn.norm_q,
            attn.norm_k,
            pack_flux3_rotary_emb(image_rotary_emb),
            output=(query, key, value),
            attn_tokens=num_tokens,
        )
        attention_output = torch.empty(
            batch_size, num_tokens_pad, channels, dtype=hidden_states.dtype, device=hidden_states.device
        )
        attention_fp16(query, key, value, attention_output, attn.head_dim ** (-0.5))

        hidden_states = attention_output[:, :num_tokens]
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

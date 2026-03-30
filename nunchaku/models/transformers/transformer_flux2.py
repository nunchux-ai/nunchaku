"""
Python-only Nunchaku runtime for FLUX.2 transformers.
"""

import gc
import json
import math
from pathlib import Path
import os
import torch
from warnings import warn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
)
from huggingface_hub import utils

try:
    from diffusers.utils import apply_lora_scale
except ImportError:

    def apply_lora_scale(_kwargs_name: str = "joint_attention_kwargs"):
        def decorator(func):
            return func

        return decorator

from ..._C.ops import attention_fp16
from ...ops.fused import fused_qkv_norm_rottary
from ...torch_transfer_utils import pin_state_dict, resolve_pin_memory
from ...utils import (
    check_hardware_compatibility,
    get_precision,
    get_precision_from_quantization_config,
    pad_tensor,
)
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..utils import CPUOffloadManager, fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key


def _flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache=None,
    backend=None,
) -> torch.Tensor:
    if num_ref_tokens == 0 and kv_cache is None:
        return dispatch_attention_fn(query, key, value, backend=backend)

    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_all = torch.cat([key[:, :num_txt_tokens], k_ref, key[:, num_txt_tokens:]], dim=1)
        v_all = torch.cat([value[:, :num_txt_tokens], v_ref, value[:, num_txt_tokens:]], dim=1)
        return dispatch_attention_fn(query, k_all, v_all, backend=backend)

    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens

    q_txt = query[:, :ref_start]
    q_ref = query[:, ref_start:ref_end]
    q_img = query[:, ref_end:]

    k_txt = key[:, :ref_start]
    k_ref = key[:, ref_start:ref_end]
    k_img = key[:, ref_end:]

    v_txt = value[:, :ref_start]
    v_ref = value[:, ref_start:ref_end]
    v_img = value[:, ref_end:]

    q_txt_img = torch.cat([q_txt, q_img], dim=1)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=1)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=1)
    attn_txt_img = dispatch_attention_fn(query=q_txt_img, key=k_all, value=v_all, backend=backend)
    attn_txt = attn_txt_img[:, :ref_start]
    attn_img = attn_txt_img[:, ref_start:]
    attn_ref = dispatch_attention_fn(query=q_ref, key=k_ref, value=v_ref, backend=backend)
    return torch.cat([attn_txt, attn_ref, attn_img], dim=1)


def _pack_flux2_rotary_emb(freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs_cis
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("Expected Flux.2 rotary embeddings as a (cos, sin) tuple with shape (seq_len, dim).")

    # Flux.2 uses repeat_interleave_real=True, so every rotary pair shares the same cos/sin values.
    rotemb = torch.stack([sin[:, 0::2], cos[:, 0::2]], dim=-1).unsqueeze(0).unsqueeze(-2).contiguous()
    return pack_rotemb(pad_tensor(rotemb, 256, 1))


def _alloc_packed_qkv(batch_size: int, heads: int, num_tokens: int, head_dim: int, device: torch.device, pad_size: int = 256):
    num_tokens_pad = math.ceil(num_tokens / pad_size) * pad_size
    query = torch.empty(batch_size, heads, num_tokens_pad, head_dim, dtype=torch.float16, device=device)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    return query, key, value, num_tokens_pad


def _apply_gated_residual(residual: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled():
        return residual + gate * update
    residual.addcmul_(gate, update)
    return residual


class NunchakuFlux2Attention(Flux2Attention):
    def __init__(self, other: Flux2Attention, **kwargs):
        super(Flux2Attention, self).__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias
        self.fused_projections = True
        processor = getattr(other, "processor", None)
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.to_out = other.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k
            self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)
            with torch.device("meta"):
                to_added_qkv = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
            self.to_added_qkv = SVDQW4A4Linear.from_linear(to_added_qkv, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache", None)
        kv_cache_mode = kwargs.get("kv_cache_mode", None)
        num_ref_tokens = int(kwargs.get("num_ref_tokens", 0))
        use_packed_fp16 = (
            kv_cache_mode is None
            and encoder_hidden_states is not None
            and isinstance(image_rotary_emb, tuple)
            and len(image_rotary_emb) == 2
            and image_rotary_emb[0].ndim == 3
            and hidden_states.is_cuda
        )
        if use_packed_fp16:
            batch_size = hidden_states.shape[0]
            num_txt_tokens = encoder_hidden_states.shape[1]
            num_img_tokens = hidden_states.shape[1]
            num_txt_tokens_pad = math.ceil(num_txt_tokens / 256) * 256
            num_img_tokens_pad = math.ceil(num_img_tokens / 256) * 256
            num_tokens_pad = num_txt_tokens_pad + num_img_tokens_pad
            query = torch.empty(
                batch_size, self.heads, num_tokens_pad, self.head_dim, dtype=torch.float16, device=hidden_states.device
            )
            key = torch.empty_like(query)
            value = torch.empty_like(query)
            fused_qkv_norm_rottary(
                hidden_states,
                self.to_qkv,
                self.norm_q,
                self.norm_k,
                image_rotary_emb[0],
                output=(
                    query[:, :, num_txt_tokens_pad:],
                    key[:, :, num_txt_tokens_pad:],
                    value[:, :, num_txt_tokens_pad:],
                ),
                attn_tokens=num_img_tokens,
            )
            fused_qkv_norm_rottary(
                encoder_hidden_states,
                self.to_added_qkv,
                self.norm_added_q,
                self.norm_added_k,
                image_rotary_emb[1],
                output=(query[:, :, :num_txt_tokens_pad], key[:, :, :num_txt_tokens_pad], value[:, :, :num_txt_tokens_pad]),
                attn_tokens=num_txt_tokens,
            )
            attention_output = torch.empty(
                batch_size,
                num_tokens_pad,
                self.heads * self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_fp16(query, key, value, attention_output, self.head_dim ** (-0.5))
            encoder_hidden_states = attention_output[:, :num_txt_tokens]
            hidden_states = attention_output[:, num_txt_tokens_pad : num_txt_tokens_pad + num_img_tokens]
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states, encoder_hidden_states

        if (
            encoder_hidden_states is not None
            and isinstance(image_rotary_emb, tuple)
            and len(image_rotary_emb) == 2
            and image_rotary_emb[0].ndim == 3
        ):
            batch_size = hidden_states.shape[0]
            qkv = fused_qkv_norm_rottary(
                hidden_states,
                self.to_qkv,
                self.norm_q,
                self.norm_k,
                image_rotary_emb[0],
            )
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(batch_size, -1, self.heads, self.head_dim)
            key = key.view(batch_size, -1, self.heads, self.head_dim)
            value = value.view(batch_size, -1, self.heads, self.head_dim)

            encoder_qkv = fused_qkv_norm_rottary(
                encoder_hidden_states,
                self.to_added_qkv,
                self.norm_added_q,
                self.norm_added_k,
                image_rotary_emb[1],
            )
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)
            encoder_query = encoder_query.view(batch_size, -1, self.heads, self.head_dim)
            encoder_key = encoder_key.view(batch_size, -1, self.heads, self.head_dim)
            encoder_value = encoder_value.view(batch_size, -1, self.heads, self.head_dim)
            encoder_seq_len = encoder_hidden_states.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        else:
            query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)
            query = query.unflatten(-1, (self.heads, -1))
            key = key.unflatten(-1, (self.heads, -1))
            value = value.unflatten(-1, (self.heads, -1))
            query = self.norm_q(query)
            key = self.norm_k(key)

            encoder_seq_len = 0
            if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
                encoder_query, encoder_key, encoder_value = self.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
                encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (self.heads, -1))
                encoder_query = self.norm_added_q(encoder_query)
                encoder_key = self.norm_added_k(encoder_key)
                encoder_seq_len = encoder_hidden_states.shape[1]
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = encoder_seq_len
            ref_end = encoder_seq_len + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, encoder_seq_len, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, encoder_seq_len, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_seq_len:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len], dim=1
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if encoder_seq_len:
            return hidden_states, encoder_hidden_states
        return hidden_states


class NunchakuFlux2FeedForward(Flux2FeedForward):
    def __init__(self, other: Flux2FeedForward, **kwargs):
        super(Flux2FeedForward, self).__init__()
        self.linear_in = SVDQW4A4Linear.from_linear(other.linear_in, **kwargs)
        self.act_fn = other.act_fn
        self.linear_out = SVDQW4A4Linear.from_linear(other.linear_out, **kwargs)
        # FLUX.2 PTQ does not currently apply ShiftedLinear on these SwiGLU down-projections,
        # so int4 must keep the signed activation path.
        self.linear_out.act_unsigned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class NunchakuFlux2ParallelSelfAttention(Flux2ParallelSelfAttention):
    def __init__(self, other: Flux2ParallelSelfAttention, **kwargs):
        super(Flux2ParallelSelfAttention, self).__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.mlp_ratio = other.mlp_ratio
        self.mlp_hidden_dim = other.mlp_hidden_dim
        self.mlp_mult_factor = other.mlp_mult_factor
        processor = getattr(other, "processor", None)
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

        # Keep clear parameter names for export/runtime alignment.
        with torch.device("meta"):
            qkv_proj = torch.nn.Linear(other.query_dim, other.inner_dim * 3, bias=other.use_bias)
            mlp_fc1 = torch.nn.Linear(other.query_dim, other.mlp_hidden_dim * other.mlp_mult_factor, bias=other.use_bias)
            out_proj = torch.nn.Linear(other.inner_dim, other.out_dim, bias=other.to_out.bias is not None)
            mlp_fc2 = torch.nn.Linear(other.mlp_hidden_dim, other.out_dim, bias=other.to_out.bias is not None)
        self.qkv_proj = SVDQW4A4Linear.from_linear(qkv_proj, **kwargs)
        self.mlp_fc1 = SVDQW4A4Linear.from_linear(mlp_fc1, **kwargs)
        self.mlp_act_fn = other.mlp_act_fn
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.out_proj = SVDQW4A4Linear.from_linear(out_proj, **kwargs)
        self.mlp_fc2 = SVDQW4A4Linear.from_linear(mlp_fc2, **kwargs)
        # FLUX.2 PTQ does not currently apply ShiftedLinear on these SwiGLU down-projections,
        # so int4 must keep the signed activation path.
        self.mlp_fc2.act_unsigned = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache", None)
        kv_cache_mode = kwargs.get("kv_cache_mode", None)
        num_txt_tokens = int(kwargs.get("num_txt_tokens", 0))
        num_ref_tokens = int(kwargs.get("num_ref_tokens", 0))
        use_packed_fp16 = kv_cache_mode is None and torch.is_tensor(image_rotary_emb) and image_rotary_emb.ndim == 3 and hidden_states.is_cuda
        if use_packed_fp16:
            batch_size = hidden_states.shape[0]
            num_tokens = hidden_states.shape[1]
            query, key, value, num_tokens_pad = _alloc_packed_qkv(
                batch_size, self.heads, num_tokens, self.head_dim, hidden_states.device
            )
            fused_qkv_norm_rottary(
                hidden_states,
                self.qkv_proj,
                self.norm_q,
                self.norm_k,
                image_rotary_emb,
                output=(query, key, value),
                attn_tokens=num_tokens,
            )
            attn_output = torch.empty(
                batch_size,
                num_tokens_pad,
                self.heads * self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_fp16(query, key, value, attn_output, self.head_dim ** (-0.5))
            attn_output = attn_output[:, :num_tokens]
            mlp_hidden_states = self.mlp_act_fn(self.mlp_fc1(hidden_states))
            return self.out_proj(attn_output) + self.mlp_fc2(mlp_hidden_states)

        if torch.is_tensor(image_rotary_emb) and image_rotary_emb.ndim == 3:
            batch_size = hidden_states.shape[0]
            qkv = fused_qkv_norm_rottary(hidden_states, self.qkv_proj, self.norm_q, self.norm_k, image_rotary_emb)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(batch_size, -1, self.heads, self.head_dim)
            key = key.view(batch_size, -1, self.heads, self.head_dim)
            value = value.view(batch_size, -1, self.heads, self.head_dim)
        else:
            qkv = self.qkv_proj(hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (self.heads, -1))
            key = key.unflatten(-1, (self.heads, -1))
            value = value.unflatten(-1, (self.heads, -1))
            query = self.norm_q(query)
            key = self.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            attn_output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = self.mlp_act_fn(self.mlp_fc1(hidden_states))
        return self.out_proj(attn_output) + self.mlp_fc2(mlp_hidden_states)


class NunchakuFlux2TransformerBlock(Flux2TransformerBlock):
    def __init__(self, block: Flux2TransformerBlock, **kwargs):
        super(Flux2TransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context
        self.attn = NunchakuFlux2Attention(block.attn, **kwargs)
        self.norm2 = block.norm2
        self.ff = NunchakuFlux2FeedForward(block.ff, **kwargs)
        self.norm2_context = block.norm2_context
        self.ff_context = NunchakuFlux2FeedForward(block.ff_context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = _apply_gated_residual(hidden_states, gate_msa, attn_output)

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        hidden_states = _apply_gated_residual(hidden_states, gate_mlp, self.ff(norm_hidden_states))

        encoder_hidden_states = _apply_gated_residual(encoder_hidden_states, c_gate_msa, context_attn_output)

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = _apply_gated_residual(
            encoder_hidden_states, c_gate_mlp, self.ff_context(norm_encoder_hidden_states)
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):
    def __init__(self, block: Flux2SingleTransformerBlock, **kwargs):
        super(Flux2SingleTransformerBlock, self).__init__()
        self.norm = block.norm
        self.attn = NunchakuFlux2ParallelSelfAttention(block.attn, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = _apply_gated_residual(hidden_states, mod_gate, attn_output)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        return hidden_states


class NunchakuFlux2Transformer2DModel(Flux2Transformer2DModel, NunchakuModelLoaderMixin):
    def _patch_model(self, **kwargs):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFlux2TransformerBlock(block, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFlux2SingleTransformerBlock(block, **kwargs)
        self.offload = False
        self.transformer_block_offload_manager = None
        self.single_transformer_block_offload_manager = None
        return self

    def set_offload(self, offload: bool, **kwargs):
        if offload == self.offload:
            return

        self.offload = offload
        if offload:
            use_pin_memory = kwargs.get("use_pin_memory", True)
            num_blocks_on_gpu = kwargs.get("num_blocks_on_gpu", 1)
            self.transformer_block_offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=use_pin_memory,
                on_gpu_modules=[
                    self.time_guidance_embed,
                    self.double_stream_modulation_img,
                    self.double_stream_modulation_txt,
                    self.single_stream_modulation,
                    self.x_embedder,
                    self.context_embedder,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=num_blocks_on_gpu,
            )
            self.single_transformer_block_offload_manager = CPUOffloadManager(
                self.single_transformer_blocks,
                use_pin_memory=use_pin_memory,
                num_blocks_on_gpu=num_blocks_on_gpu,
            )
        else:
            self.transformer_block_offload_manager = None
            self.single_transformer_block_offload_manager = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict | None = None,
        return_dict: bool = True,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if kv_cache_mode is not None:
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                kv_cache=kv_cache,
                kv_cache_mode=kv_cache_mode,
                num_ref_tokens=num_ref_tokens,
                ref_fixed_timestep=ref_fixed_timestep,
            )

        if self.offload:
            device = hidden_states.device
            self.transformer_block_offload_manager.set_device(device)
            self.single_transformer_block_offload_manager.set_device(device)

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        rotary_emb_img = _pack_flux2_rotary_emb(image_rotary_emb)
        rotary_emb_txt = _pack_flux2_rotary_emb(text_rotary_emb)
        rotary_emb_single = _pack_flux2_rotary_emb(
            (
                torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )
        )
        kv_attn_kwargs = joint_attention_kwargs

        if self.offload:
            compute_stream = torch.cuda.current_stream()
            self.transformer_block_offload_manager.initialize(compute_stream)
            for index_block in range(len(self.transformer_blocks)):
                with torch.cuda.stream(compute_stream):
                    block = self.transformer_block_offload_manager.get_block(index_block)
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                            block,
                            hidden_states,
                            encoder_hidden_states,
                            double_stream_mod_img,
                            double_stream_mod_txt,
                            (rotary_emb_img, rotary_emb_txt),
                            kv_attn_kwargs,
                        )
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb_mod_img=double_stream_mod_img,
                            temb_mod_txt=double_stream_mod_txt,
                            image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                            joint_attention_kwargs=kv_attn_kwargs,
                        )
                self.transformer_block_offload_manager.step(compute_stream)
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        double_stream_mod_img,
                        double_stream_mod_txt,
                        (rotary_emb_img, rotary_emb_txt),
                        kv_attn_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb_mod_img=double_stream_mod_img,
                        temb_mod_txt=double_stream_mod_txt,
                        image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                        joint_attention_kwargs=kv_attn_kwargs,
                    )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        kv_attn_kwargs_single = kv_attn_kwargs

        if self.offload:
            self.single_transformer_block_offload_manager.initialize(compute_stream)
            for index_block in range(len(self.single_transformer_blocks)):
                with torch.cuda.stream(compute_stream):
                    block = self.single_transformer_block_offload_manager.get_block(index_block)
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        hidden_states = self._gradient_checkpointing_func(
                            block,
                            hidden_states,
                            None,
                            single_stream_mod,
                            rotary_emb_single,
                            kv_attn_kwargs_single,
                        )
                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=None,
                            temb_mod=single_stream_mod,
                            image_rotary_emb=rotary_emb_single,
                            joint_attention_kwargs=kv_attn_kwargs_single,
                        )
                self.single_transformer_block_offload_manager.step(compute_stream)
        else:
            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        None,
                        single_stream_mod,
                        rotary_emb_single,
                        kv_attn_kwargs_single,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=None,
                        temb_mod=single_stream_mod,
                        image_rotary_emb=rotary_emb_single,
                        joint_attention_kwargs=kv_attn_kwargs_single,
                    )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)
        pin_memory = kwargs.get("pin_memory", "auto")
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if offload:
            raise NotImplementedError("Offload is not supported for NunchakuFlux2Transformer2DModel")

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        if not (
            pretrained_model_name_or_path.is_file()
            or pretrained_model_name_or_path.name.endswith((".safetensors", ".sft"))
        ):
            raise AssertionError("Only safetensors are supported")

        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = int(quantization_config.get("rank", 32))
        if quantization_config:
            precision = get_precision_from_quantization_config(quantization_config)
            if torch.device(device).type == "cuda":
                check_hardware_compatibility(quantization_config, device)
        else:
            precision = get_precision(device=device)
            if precision == "fp4":
                precision = "nvfp4"

        transformer = transformer.to(torch_dtype)
        transformer._patch_model(precision=precision, rank=rank, torch_dtype=torch_dtype)
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        if resolve_pin_memory(pin_memory, device):
            model_state_dict = pin_state_dict(model_state_dict)

        transformer.load_state_dict(model_state_dict)

        if kwargs.get("return_metadata", False):
            return transformer, metadata
        return transformer

    def to(self, *args, **kwargs):
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs

        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if getattr(self, "offload", False) and device_arg_or_kwarg_present:
            warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
            return self
        return super(type(self), self).to(*args, **kwargs)

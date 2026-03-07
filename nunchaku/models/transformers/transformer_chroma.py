from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


# C++ additive attention (rank-1 bias mask) is enabled by default.
# Set `NUNCHAKU_CHROMA_USE_CPP_ADDITIVE_ATTN=0` to force the Python/SDPA path.
_CPP_ADDITIVE_ATTN_DEFAULT = "1"
_CPP_ADDITIVE_ATTN_FALLBACK_LOGGED = False


def _log_cpp_additive_attn_fallback(reason: str):
    global _CPP_ADDITIVE_ATTN_FALLBACK_LOGGED
    if _CPP_ADDITIVE_ATTN_FALLBACK_LOGGED:
        return
    _CPP_ADDITIVE_ATTN_FALLBACK_LOGGED = True
    print(f"[nunchaku.chroma] cpp_additive_attn fallback: {reason}")


@dataclass(frozen=True)
class LoadReport:
    config: dict[str, Any]
    precision: str
    rank: int


def _maybe_log(verbose: bool, *args):
    if verbose:
        print(*args)


def _load_safetensors_state_dict(
    path: str | Path,
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    from nunchaku.utils import load_state_dict_in_safetensors

    sd, md = load_state_dict_in_safetensors(path, device=device, return_metadata=True)
    return sd, md


def _convert_checkpoint_key_for_svdq_linear(k: str) -> str:
    # This safetensors uses nunchaku converter naming:
    # - lora_down/lora_up are actually SVD proj_down/proj_up
    # - smooth/smooth_orig are smooth_factor/smooth_factor_orig
    if ".lora_down" in k:
        k = k.replace(".lora_down", ".proj_down")
    if ".lora_up" in k:
        k = k.replace(".lora_up", ".proj_up")
    if ".smooth_orig" in k:
        k = k.replace(".smooth_orig", ".smooth_factor_orig")
    elif ".smooth" in k:
        k = k.replace(".smooth", ".smooth_factor")
    return k


def _convert_checkpoint_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    return {_convert_checkpoint_key_for_svdq_linear(k): v for k, v in sd.items()}


def _infer_rank_from_converted_state_dict(sd: dict[str, Any]) -> int:
    # Look for any SVDQW4A4Linear proj_down shaped [in_features, rank]
    for k, v in sd.items():
        if k.endswith(".proj_down") and getattr(v, "ndim", None) == 2:
            return int(v.shape[1])
    raise ValueError("Cannot infer SVD rank from checkpoint (missing any '*.proj_down' tensors).")


def _build_attn_norms(*, head_dim: int, eps: float, with_added: bool, device, dtype):
    from diffusers.models.normalization import RMSNorm

    m = nn.Module()
    m.norm_q = RMSNorm(head_dim, eps=eps, elementwise_affine=True).to(device=device, dtype=dtype)
    m.norm_k = RMSNorm(head_dim, eps=eps, elementwise_affine=True).to(device=device, dtype=dtype)
    if with_added:
        m.norm_added_q = RMSNorm(head_dim, eps=eps, elementwise_affine=True).to(device=device, dtype=dtype)
        m.norm_added_k = RMSNorm(head_dim, eps=eps, elementwise_affine=True).to(device=device, dtype=dtype)
    return m


def _dispatch_attention(
    query,
    key,
    value,
    attention_mask,
    *,
    txt_tokens: int | None = None,
    valid_txt: int | None = None,
):
    """
    Chroma attention dispatch.

    Performance note:
    This function must NOT call `.item()` on CUDA tensors (it would introduce a device sync per block).
    Any fast-path metadata (e.g. `valid_txt`) should be computed once per model forward.
    """
    del txt_tokens, valid_txt

    from diffusers.models.transformers.transformer_flux import dispatch_attention_fn

    # No mask: allow fastest backend selection (FLASH where available).
    if attention_mask is None:
        return dispatch_attention_fn(query, key, value, attn_mask=None, backend=None)

    # Speed + quality path (Chroma-specific):
    # The Chroma pipeline provides a 2D mask `m` (values in {0,1}, dtype usually bf16/fp16), which diffusers expands
    # to a rank-1 outer-product bias `m_i * m_j` and passes as an additive SDPA mask.
    # This is *not* a boolean hard-mask, but an additive bias in SDPA.
    #
    # We can fold this outer-product bias into the QK dot-product by augmenting Q/K with extra dims, and then run
    # fast attention with attn_mask=None while preserving semantics closely.
    if attention_mask.ndim == 2 and query.shape[0] == 1:
        b, s = attention_mask.shape
        if b != 1:
            raise ValueError(f"Only batch_size=1 is supported for folded-mask fast path (got B={b}).")
        if int(query.shape[1]) != int(s) or int(key.shape[1]) != int(s):
            raise ValueError(
                f"Mask/sequence length mismatch: mask S={int(s)}, query S={int(query.shape[1])}, key S={int(key.shape[1])}"
            )

        # Expand to (B,S,H,1) and keep dtype aligned with Q/K.
        m1 = attention_mask.to(dtype=query.dtype)[:, :, None, None].expand(
            query.shape[0], query.shape[1], query.shape[2], 1
        )

        d = int(query.shape[-1])
        scale = float(d) ** -0.5

        # Keep extra dims minimal but aligned (multiple of 8) to reduce overhead.
        extra = 8
        sqrt_d = float(d) ** 0.5

        q_extra = torch.cat([m1 * sqrt_d, m1.new_zeros((*m1.shape[:-1], extra - 1))], dim=-1)
        k_extra = torch.cat([m1, m1.new_zeros((*m1.shape[:-1], extra - 1))], dim=-1)
        v_extra = value.new_zeros((*value.shape[:-1], extra))

        q_ext = torch.cat([query, q_extra], dim=-1)
        k_ext = torch.cat([key, k_extra], dim=-1)
        v_ext = torch.cat([value, v_extra], dim=-1)

        # Prefer native flash kernel when available; pass explicit scale to preserve original head_dim scaling.
        try:
            out_ext = dispatch_attention_fn(q_ext, k_ext, v_ext, attn_mask=None, backend="_native_flash", scale=scale)
        except TypeError:
            # Older diffusers may not expose `scale` in dispatch_attention_fn; fallback to correctness baseline.
            out_ext = None
        if out_ext is not None:
            return out_ext[..., :d]

    # Fallback: preserve diffusers Chroma mask semantics (outer-product additive bias) and use SDPA efficient.
    attn_mask_4d = NunchakuChromaTransformerBlockMixin._mask_to_4d(attention_mask)
    return dispatch_attention_fn(query, key, value, attn_mask=attn_mask_4d, backend="_native_efficient")


class NunchakuChromaTransformerBlockMixin:
    @staticmethod
    def _mask_to_4d(attention_mask):
        # Match diffusers `transformer_chroma` behavior:
        # Expand a 2D mask to a full QK mask (outer product).
        #
        # IMPORTANT: do NOT cast to bool here. Chroma's pipeline may provide a non-bool mask (e.g. bf16 0/1),
        # and changing dtype/value semantics affects output quality.
        if attention_mask is None:
            return None
        if attention_mask.ndim == 4:
            return attention_mask
        if attention_mask.ndim != 2:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")
        return attention_mask[:, None, None, :] * attention_mask[:, None, :, None]


class NunchakuChromaSingleTransformerBlock(nn.Module, NunchakuChromaTransformerBlockMixin):
    """
    Matches the checkpoint key layout under:
      single_transformer_blocks.<i>.{qkv_proj,out_proj,mlp_fc1,mlp_fc2,attn.norm_{q,k},norm,proj_out?}
    """

    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        rank: int,
        precision: str,
        device,
        dtype,
        eps: float = 1e-6,
    ):
        super().__init__()
        from diffusers.models.transformers.transformer_chroma import ChromaAdaLayerNormZeroSinglePruned
        from nunchaku.models.linear import SVDQW4A4Linear

        self.heads = int(num_attention_heads)
        self.head_dim = int(attention_head_dim)
        self.inner_dim = int(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = ChromaAdaLayerNormZeroSinglePruned(dim).to(device=device, dtype=dtype)
        self.attn = _build_attn_norms(head_dim=self.head_dim, eps=eps, with_added=False, device=device, dtype=dtype)

        self.qkv_proj = SVDQW4A4Linear(
            in_features=dim,
            out_features=3 * dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.out_proj = SVDQW4A4Linear(
            in_features=dim,
            out_features=dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.mlp_fc1 = SVDQW4A4Linear(
            in_features=dim,
            out_features=self.mlp_hidden_dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.mlp_fc2 = SVDQW4A4Linear(
            in_features=self.mlp_hidden_dim,
            out_features=dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        # Temporary diagnostic workaround:
        # Chroma int4 down-proj parity is badly broken when using the unsigned activation path.
        # Force the signed path here to validate runtime correctness end-to-end.
        self.mlp_fc2.act_unsigned = False

        self.norm_q = self.attn.norm_q
        self.norm_k = self.attn.norm_k

    def forward(
        self,
        hidden_states,
        temb,
        image_rotary_emb=None,
        attention_mask=None,
        attention_mask_1d=None,
        txt_tokens: int | None = None,
        valid_txt: int | None = None,
        joint_attention_kwargs=None,
        cpp_workspace: dict | None = None,
        cpp_mask: torch.Tensor | None = None,
    ):
        from nunchaku.ops.fused import fused_gelu_mlp, fused_qkv_norm_rottary

        if joint_attention_kwargs is not None and len(joint_attention_kwargs) > 0:
            raise NotImplementedError("joint_attention_kwargs is not supported in NunchakuChromaSingleTransformerBlock")

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        mlp_out = fused_gelu_mlp(norm_hidden_states, self.mlp_fc1, self.mlp_fc2)

        # Optional C++/CUDA additive attention backend (exact Chroma semantics, B=1 only).
        use_cpp_requested = os.getenv("NUNCHAKU_CHROMA_USE_CPP_ADDITIVE_ATTN", _CPP_ADDITIVE_ATTN_DEFAULT) == "1"
        use_cpp = (
            use_cpp_requested
            and attention_mask_1d is not None
            and norm_hidden_states.is_cuda
            and int(self.head_dim) == 128
        )
        if use_cpp_requested and not use_cpp:
            if attention_mask_1d is None:
                _log_cpp_additive_attn_fallback("missing attention_mask (attention_mask_1d is None)")
            elif not norm_hidden_states.is_cuda:
                _log_cpp_additive_attn_fallback("hidden_states is not CUDA")
            elif int(self.head_dim) != 128:
                _log_cpp_additive_attn_fallback(f"unsupported head_dim={int(self.head_dim)} (expected 128)")
        if use_cpp:
            try:
                from nunchaku._C.ops import chroma_additive_attention_packed_fp16

                pad_size = 256
                b, s = int(norm_hidden_states.shape[0]), int(norm_hidden_states.shape[1])
                s_pad = int(math.ceil(s / pad_size) * pad_size)

                if cpp_workspace is None:
                    q = torch.empty(
                        (b, self.heads, s_pad, self.head_dim),
                        device=norm_hidden_states.device,
                        dtype=torch.float16,
                    )
                    k = torch.empty_like(q)
                    v = torch.empty_like(q)
                    m = torch.empty((b, s_pad), device=norm_hidden_states.device, dtype=torch.float16)
                    out = torch.empty(
                        (b, s_pad, self.heads * self.head_dim),
                        device=norm_hidden_states.device,
                        dtype=norm_hidden_states.dtype,
                    )
                else:
                    q, k, v, m, out = (
                        cpp_workspace["q"],
                        cpp_workspace["k"],
                        cpp_workspace["v"],
                        cpp_workspace["m"],
                        cpp_workspace["out"],
                    )

                fused_qkv_norm_rottary(
                    norm_hidden_states,
                    self.qkv_proj,
                    self.attn.norm_q,
                    self.attn.norm_k,
                    image_rotary_emb,
                    output=(q, k, v),
                    attn_tokens=s,
                )

                if cpp_mask is not None:
                    m = cpp_mask
                else:
                    m.zero_()
                    m[:, :s] = attention_mask_1d.to(dtype=torch.float16)

                chroma_additive_attention_packed_fp16(q, k, v, m, out, 0.0)
                attn_out = out[:, :s, :]
            except Exception as e:
                _log_cpp_additive_attn_fallback(f"exception in single-block cpp path: {type(e).__name__}: {e}")
                use_cpp = False

        if not use_cpp:
            qkv = fused_qkv_norm_rottary(
                norm_hidden_states,
                self.qkv_proj,
                self.attn.norm_q,
                self.attn.norm_k,
                image_rotary_emb,
            )
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (self.heads, -1))
            key = key.unflatten(-1, (self.heads, -1))
            value = value.unflatten(-1, (self.heads, -1))

            attn_mask = attention_mask_1d if attention_mask_1d is not None else self._mask_to_4d(attention_mask)
            attn_out = _dispatch_attention(query, key, value, attn_mask, txt_tokens=txt_tokens, valid_txt=valid_txt)
            attn_out = attn_out.flatten(2, 3).to(query.dtype)

        proj = self.out_proj(attn_out) + mlp_out
        hidden_states = residual + gate.unsqueeze(1) * proj

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class NunchakuChromaTransformerBlock(nn.Module, NunchakuChromaTransformerBlockMixin):
    """
    Matches the checkpoint key layout under:
      transformer_blocks.<i>.{qkv_proj,qkv_proj_context,out_proj,out_proj_context,mlp_fc1,mlp_fc2,mlp_context_fc1,mlp_context_fc2,attn.*}
    """

    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        rank: int,
        precision: str,
        device,
        dtype,
        eps: float = 1e-6,
    ):
        super().__init__()
        from diffusers.models.transformers.transformer_chroma import ChromaAdaLayerNormZeroPruned
        from nunchaku.models.linear import SVDQW4A4Linear

        self.heads = int(num_attention_heads)
        self.head_dim = int(attention_head_dim)
        self.inner_dim = int(dim)

        self.norm1 = ChromaAdaLayerNormZeroPruned(dim).to(device=device, dtype=dtype)
        self.norm1_context = ChromaAdaLayerNormZeroPruned(dim).to(device=device, dtype=dtype)

        self.attn = _build_attn_norms(head_dim=self.head_dim, eps=eps, with_added=True, device=device, dtype=dtype)

        self.qkv_proj = SVDQW4A4Linear(
            in_features=dim,
            out_features=3 * dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.qkv_proj_context = SVDQW4A4Linear(
            in_features=dim,
            out_features=3 * dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.out_proj = SVDQW4A4Linear(
            in_features=dim,
            out_features=dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.out_proj_context = SVDQW4A4Linear(
            in_features=dim,
            out_features=dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype)

        self.mlp_fc1 = SVDQW4A4Linear(
            in_features=dim,
            out_features=4 * dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.mlp_fc2 = SVDQW4A4Linear(
            in_features=4 * dim,
            out_features=dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        # Temporary diagnostic workaround:
        # Chroma int4 down-proj parity is badly broken when using the unsigned activation path.
        # Force the signed path here to validate runtime correctness end-to-end.
        self.mlp_fc2.act_unsigned = False

        self.mlp_context_fc1 = SVDQW4A4Linear(
            in_features=dim,
            out_features=4 * dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        self.mlp_context_fc2 = SVDQW4A4Linear(
            in_features=4 * dim,
            out_features=dim,
            rank=rank,
            bias=True,
            precision=precision,
            torch_dtype=dtype,
            device=device,
        )
        # Temporary diagnostic workaround:
        # Chroma int4 down-proj parity is badly broken when using the unsigned activation path.
        # Force the signed path here to validate runtime correctness end-to-end.
        self.mlp_context_fc2.act_unsigned = False

        self.act = nn.GELU(approximate="tanh")

        self.norm_q = self.attn.norm_q
        self.norm_k = self.attn.norm_k
        self.norm_added_q = self.attn.norm_added_q
        self.norm_added_k = self.attn.norm_added_k

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb=None,
        attention_mask=None,
        valid_txt: int | None = None,
        joint_attention_kwargs=None,
        attention_mask_1d=None,
        cpp_workspace: dict | None = None,
        cpp_mask: torch.Tensor | None = None,
    ):
        from nunchaku.ops.fused import fused_gelu_mlp, fused_qkv_norm_rottary

        if joint_attention_kwargs is not None and len(joint_attention_kwargs) > 0:
            raise NotImplementedError("joint_attention_kwargs is not supported in NunchakuChromaTransformerBlock")

        temb_img, temb_txt = temb[:, :6], temb[:, 6:]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb_img)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb_txt
        )

        rotary_img, rotary_txt = image_rotary_emb
        use_cpp_requested = os.getenv("NUNCHAKU_CHROMA_USE_CPP_ADDITIVE_ATTN", _CPP_ADDITIVE_ATTN_DEFAULT) == "1"
        use_cpp = (
            use_cpp_requested
            and attention_mask_1d is not None
            and norm_hidden_states.is_cuda
            and int(self.head_dim) == 128
        )
        if use_cpp_requested and not use_cpp:
            if attention_mask_1d is None:
                _log_cpp_additive_attn_fallback("missing attention_mask (attention_mask_1d is None)")
            elif not norm_hidden_states.is_cuda:
                _log_cpp_additive_attn_fallback("hidden_states is not CUDA")
            elif int(self.head_dim) != 128:
                _log_cpp_additive_attn_fallback(f"unsupported head_dim={int(self.head_dim)} (expected 128)")

        txt_len = int(norm_encoder_hidden_states.shape[1])
        img_len = int(norm_hidden_states.shape[1])

        if use_cpp:
            try:
                from nunchaku._C.ops import chroma_additive_attention_packed_fp16

                pad_size = 256
                txt_pad = int(math.ceil(txt_len / pad_size) * pad_size)
                img_pad = int(math.ceil(img_len / pad_size) * pad_size)
                num_tokens_pad = txt_pad + img_pad
                b = int(norm_hidden_states.shape[0])

                if cpp_workspace is None:
                    q = torch.empty(
                        (b, self.heads, num_tokens_pad, self.head_dim),
                        device=norm_hidden_states.device,
                        dtype=torch.float16,
                    )
                    k = torch.empty_like(q)
                    v = torch.empty_like(q)
                    m = torch.empty((b, num_tokens_pad), device=norm_hidden_states.device, dtype=torch.float16)
                    out = torch.empty(
                        (b, num_tokens_pad, self.heads * self.head_dim),
                        device=norm_hidden_states.device,
                        dtype=norm_hidden_states.dtype,
                    )
                else:
                    q, k, v, m, out = (
                        cpp_workspace["q"],
                        cpp_workspace["k"],
                        cpp_workspace["v"],
                        cpp_workspace["m"],
                        cpp_workspace["out"],
                    )

                fused_qkv_norm_rottary(
                    norm_hidden_states,
                    self.qkv_proj,
                    self.attn.norm_q,
                    self.attn.norm_k,
                    rotary_img,
                    output=(q[:, :, txt_pad:], k[:, :, txt_pad:], v[:, :, txt_pad:]),
                    attn_tokens=img_len,
                )
                fused_qkv_norm_rottary(
                    norm_encoder_hidden_states,
                    self.qkv_proj_context,
                    self.attn.norm_added_q,
                    self.attn.norm_added_k,
                    rotary_txt,
                    output=(q[:, :, :txt_pad], k[:, :, :txt_pad], v[:, :, :txt_pad]),
                    attn_tokens=txt_len,
                )

                if cpp_mask is not None:
                    m = cpp_mask
                else:
                    m.zero_()
                    m[:, :txt_len] = attention_mask_1d[:, :txt_len].to(dtype=torch.float16)
                    m[:, txt_pad : txt_pad + img_len] = attention_mask_1d[:, txt_len : txt_len + img_len].to(
                        dtype=torch.float16
                    )

                chroma_additive_attention_packed_fp16(q, k, v, m, out, 0.0)

                context_attn_output = out[:, :txt_len, :]
                attn_output = out[:, txt_pad : txt_pad + img_len, :]
            except Exception as e:
                _log_cpp_additive_attn_fallback(f"exception in dual-block cpp path: {type(e).__name__}: {e}")
                use_cpp = False

        if not use_cpp:
            qkv = fused_qkv_norm_rottary(
                norm_hidden_states,
                self.qkv_proj,
                self.attn.norm_q,
                self.attn.norm_k,
                rotary_img,
            )
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (self.heads, -1))
            key = key.unflatten(-1, (self.heads, -1))
            value = value.unflatten(-1, (self.heads, -1))

            qkv_c = fused_qkv_norm_rottary(
                norm_encoder_hidden_states,
                self.qkv_proj_context,
                self.attn.norm_added_q,
                self.attn.norm_added_k,
                rotary_txt,
            )
            c_query, c_key, c_value = qkv_c.chunk(3, dim=-1)
            c_query = c_query.unflatten(-1, (self.heads, -1))
            c_key = c_key.unflatten(-1, (self.heads, -1))
            c_value = c_value.unflatten(-1, (self.heads, -1))

            query = torch.cat([c_query, query], dim=1)
            key = torch.cat([c_key, key], dim=1)
            value = torch.cat([c_value, value], dim=1)

            attn_mask = attention_mask_1d if attention_mask_1d is not None else self._mask_to_4d(attention_mask)
            attn_out = _dispatch_attention(query, key, value, attn_mask, txt_tokens=int(txt_len), valid_txt=valid_txt)
            attn_out = attn_out.flatten(2, 3).to(query.dtype)

            context_attn_output, attn_output = attn_out.split_with_sizes([txt_len, attn_out.shape[1] - txt_len], dim=1)

        attn_output = self.out_proj(attn_output)
        context_attn_output = self.out_proj_context(context_attn_output)

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        nh = self.norm2(hidden_states)
        nh = nh * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff = fused_gelu_mlp(nh, self.mlp_fc1, self.mlp_fc2)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        ne = self.norm2_context(encoder_hidden_states)
        ne = ne * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c_ff = fused_gelu_mlp(ne, self.mlp_context_fc1, self.mlp_context_fc2)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * c_ff

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class NunchakuChromaTransformer2dModel(ModelMixin, ConfigMixin):
    """
    A Chroma-faithful transformer that loads the exact DeepCompressor/nunchaku-ext safetensors layout.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        rank: int,
        precision: str,
        device,
        dtype,
    ):
        super().__init__()
        from diffusers.models.transformers.transformer_chroma import (
            ChromaAdaLayerNormContinuousPruned,
            ChromaApproximator,
            ChromaCombinedTimestepTextProjEmbeddings,
        )
        from nunchaku.models.embeddings import NunchakuFluxPosEmbed

        self.register_to_config(
            patch_size=int(config["patch_size"]),
            in_channels=int(config["in_channels"]),
            out_channels=config.get("out_channels", None),
            num_layers=int(config["num_layers"]),
            num_single_layers=int(config["num_single_layers"]),
            attention_head_dim=int(config["attention_head_dim"]),
            num_attention_heads=int(config["num_attention_heads"]),
            joint_attention_dim=int(config["joint_attention_dim"]),
            axes_dims_rope=tuple(config.get("axes_dims_rope", (16, 56, 56))),
            approximator_num_channels=int(config.get("approximator_num_channels", 64)),
            approximator_hidden_dim=int(config.get("approximator_hidden_dim", 5120)),
            approximator_layers=int(config.get("approximator_layers", 5)),
        )
        self.nunchaku_precision = str(precision)
        self.nunchaku_rank = int(rank)
        self.nunchaku_chroma_use_cpp_attention: bool = False
        self.nunchaku_chroma_debug: bool = False

        patch_size = int(self.config.patch_size)
        in_channels = int(self.config.in_channels)
        out_channels = int(getattr(self.config, "out_channels", None) or in_channels)
        num_layers = int(self.config.num_layers)
        num_single_layers = int(self.config.num_single_layers)
        attention_head_dim = int(self.config.attention_head_dim)
        num_attention_heads = int(self.config.num_attention_heads)
        joint_attention_dim = int(self.config.joint_attention_dim)
        axes_dims_rope = tuple(self.config.axes_dims_rope)

        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = NunchakuFluxPosEmbed(dim=self.inner_dim, theta=10000, axes_dim=list(axes_dims_rope)).to(
            device=device, dtype=dtype
        )

        self.time_text_embed = ChromaCombinedTimestepTextProjEmbeddings(
            num_channels=int(self.config.approximator_num_channels) // 4,
            out_dim=3 * num_single_layers + 2 * 6 * num_layers + 2,
        ).to(device=device, dtype=dtype)

        self.distilled_guidance_layer = ChromaApproximator(
            in_dim=int(self.config.approximator_num_channels),
            out_dim=self.inner_dim,
            hidden_dim=int(self.config.approximator_hidden_dim),
            n_layers=int(self.config.approximator_layers),
        ).to(device=device, dtype=dtype)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=True).to(device=device, dtype=dtype)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=True).to(device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                NunchakuChromaTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rank=rank,
                    precision=precision,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                NunchakuChromaSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=4.0,
                    rank=rank,
                    precision=precision,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = ChromaAdaLayerNormContinuousPruned(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        ).to(device=device, dtype=dtype)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels, bias=True).to(
            device=device, dtype=dtype
        )

        self.encoder_hid_proj = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device: str = "cuda",
        torch_dtype: Any = None,
        use_cpp_attention: bool = False,
        debug: bool = False,
        precision: str | None = None,
        rank: int | None = None,
        verbose: bool = True,
        return_report: bool = False,
    ):
        ckpt = Path(pretrained_model_name_or_path)
        if not ckpt.exists():
            raise FileNotFoundError(str(ckpt))

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        sd_raw, md = _load_safetensors_state_dict(ckpt, device="cpu")
        if "config" not in md:
            raise ValueError("Missing required safetensors metadata: 'config'")
        if "quantization_config" not in md:
            raise ValueError("Missing required safetensors metadata: 'quantization_config'")

        config = json.loads(md["config"])
        if config.get("_class_name", None) != "ChromaTransformer2DModel":
            raise ValueError(f"Unexpected config._class_name={config.get('_class_name')!r} (expected 'ChromaTransformer2DModel')")

        quant_cfg = json.loads(md["quantization_config"])
        from nunchaku.utils import get_precision_from_quantization_config

        inferred_precision = get_precision_from_quantization_config(quant_cfg)

        sd = _convert_checkpoint_state_dict(sd_raw)
        inferred_rank = _infer_rank_from_converted_state_dict(sd)

        if precision is not None and str(precision) != str(inferred_precision):
            raise ValueError(
                f"precision mismatch: got precision={precision!r}, but checkpoint says {inferred_precision!r} "
                f"(from safetensors metadata 'quantization_config')."
            )
        if rank is not None and int(rank) != int(inferred_rank):
            raise ValueError(
                f"rank mismatch: got rank={int(rank)}, but checkpoint implies rank={int(inferred_rank)} "
                f"(from '*.proj_down' tensors)."
            )

        model = cls(
            config=config,
            rank=int(inferred_rank),
            precision=str(inferred_precision),
            device=torch.device(device),
            dtype=torch_dtype,
        )
        model.nunchaku_chroma_use_cpp_attention = bool(use_cpp_attention)
        model.nunchaku_chroma_debug = bool(debug)

        from nunchaku.models.transformers.utils import patch_scale_key

        patch_scale_key(model, sd)

        wanted = set(model.state_dict().keys())
        sd_filtered = {k: v for k, v in sd.items() if k in wanted}
        model.load_state_dict(sd_filtered, strict=True)

        _maybe_log(verbose, "[nunchaku.chroma] loaded:", str(ckpt))
        _maybe_log(verbose, "[nunchaku.chroma] precision:", inferred_precision, "rank:", inferred_rank, "dtype:", torch_dtype)
        _maybe_log(
            verbose,
            "[nunchaku.chroma] config.num_layers:",
            int(config["num_layers"]),
            "num_single_layers:",
            int(config["num_single_layers"]),
        )

        if return_report:
            return model, LoadReport(config=config, precision=str(inferred_precision), rank=int(inferred_rank))
        return model

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        attention_mask=None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        del controlnet_blocks_repeat

        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from nunchaku.models.embeddings import pack_rotemb
        from nunchaku.utils import pad_tensor

        if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
            raise NotImplementedError("ControlNet is not supported in NunchakuChromaTransformer2dModel")
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            raise NotImplementedError("IP-Adapter is not supported in NunchakuChromaTransformer2dModel")

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        batch_size = int(hidden_states.shape[0])

        input_vec = self.time_text_embed(timestep)
        pooled_temb = self.distilled_guidance_layer(input_vec)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        txt_tokens = int(encoder_hidden_states.shape[1])
        img_tokens = int(hidden_states.shape[1])
        attn_mask_1d = attention_mask
        valid_txt: int | None = None
        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = pack_rotemb(pad_tensor(image_rotary_emb[:, :txt_tokens, ...], 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(image_rotary_emb[:, txt_tokens:, ...], 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(image_rotary_emb, 256, 1))

        if batch_size != int(rotary_emb_txt.shape[0]):
            rotary_emb_txt = rotary_emb_txt.expand(batch_size, -1, -1).contiguous()
        if batch_size != int(rotary_emb_img.shape[0]):
            rotary_emb_img = rotary_emb_img.expand(batch_size, -1, -1).contiguous()
        if batch_size != int(rotary_emb_single.shape[0]):
            rotary_emb_single = rotary_emb_single.expand(batch_size, -1, -1).contiguous()

        use_cpp_ws = (
            os.getenv("NUNCHAKU_CHROMA_USE_CPP_ADDITIVE_ATTN", _CPP_ADDITIVE_ATTN_DEFAULT) == "1"
            and hidden_states.is_cuda
            and attention_mask is not None
            and int(self.config.attention_head_dim) == 128
        )
        ws_dual: dict | None = None
        ws_single: dict | None = None
        mask_dual: torch.Tensor | None = None
        mask_single: torch.Tensor | None = None
        if use_cpp_ws:
            heads = int(self.config.num_attention_heads)
            head_dim = int(self.config.attention_head_dim)

            pad_size = 256
            txt_pad = int(math.ceil(txt_tokens / pad_size) * pad_size)
            img_pad = int(math.ceil(img_tokens / pad_size) * pad_size)
            num_tokens_pad = int(txt_pad + img_pad)

            key_dual = (batch_size, num_tokens_pad, heads, head_dim, str(hidden_states.device), hidden_states.dtype)
            ws_dual = getattr(self, "_nunchaku_cpp_ws_dual_shared", None)
            if ws_dual is None or ws_dual.get("key") != key_dual:
                ws_dual = {
                    "key": key_dual,
                    "q": torch.empty((batch_size, heads, num_tokens_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "k": torch.empty((batch_size, heads, num_tokens_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "v": torch.empty((batch_size, heads, num_tokens_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "m": torch.empty((batch_size, num_tokens_pad), device=hidden_states.device, dtype=torch.float16),
                    "out": torch.empty((batch_size, num_tokens_pad, heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype),
                }
                self._nunchaku_cpp_ws_dual_shared = ws_dual

            s_total = int(txt_tokens + img_tokens)
            s_pad = int(math.ceil(s_total / pad_size) * pad_size)
            key_single = (batch_size, s_pad, heads, head_dim, str(hidden_states.device), hidden_states.dtype)
            ws_single = getattr(self, "_nunchaku_cpp_ws_single_shared", None)
            if ws_single is None or ws_single.get("key") != key_single:
                ws_single = {
                    "key": key_single,
                    "q": torch.empty((batch_size, heads, s_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "k": torch.empty((batch_size, heads, s_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "v": torch.empty((batch_size, heads, s_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "m": torch.empty((batch_size, s_pad), device=hidden_states.device, dtype=torch.float16),
                    "out": torch.empty((batch_size, s_pad, heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype),
                }
                self._nunchaku_cpp_ws_single_shared = ws_single

            if attention_mask is not None:
                attn_mask_fp16 = attention_mask.to(dtype=torch.float16)
                mask_single = ws_single["m"]
                mask_single.zero_()
                mask_single[:, :s_total] = attn_mask_fp16

                mask_dual = ws_dual["m"]
                mask_dual.zero_()
                mask_dual[:, :txt_tokens] = attn_mask_fp16[:, :txt_tokens]
                mask_dual[:, txt_pad : txt_pad + img_tokens] = attn_mask_fp16[:, txt_tokens : txt_tokens + img_tokens]

        num_layers = len(self.transformer_blocks)
        num_single = len(self.single_transformer_blocks)
        img_offset = 3 * num_single
        txt_offset = img_offset + 6 * num_layers

        for i, block in enumerate(self.transformer_blocks):
            img_mod = img_offset + 6 * i
            txt_mod = txt_offset + 6 * i
            temb = torch.cat(
                (pooled_temb[:, img_mod : img_mod + 6], pooled_temb[:, txt_mod : txt_mod + 6]),
                dim=1,
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                attention_mask=None,
                valid_txt=valid_txt,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask_1d=attn_mask_1d,
                cpp_workspace=ws_dual,
                cpp_mask=mask_dual,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for i, block in enumerate(self.single_transformer_blocks):
            start = 3 * i
            temb = pooled_temb[:, start : start + 3]
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                attention_mask=None,
                attention_mask_1d=attn_mask_1d,
                txt_tokens=txt_tokens,
                valid_txt=valid_txt,
                joint_attention_kwargs=joint_attention_kwargs,
                cpp_workspace=ws_single,
                cpp_mask=mask_single,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        temb = pooled_temb[:, -2:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

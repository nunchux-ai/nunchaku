from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from warnings import warn

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


@dataclass(frozen=True)
class LoadReport:
    config: dict[str, Any]
    precision: str
    rank: int


# ---------------------------------------------------------------------------
# Checkpoint key conversion
# ---------------------------------------------------------------------------

_KEY_RENAMES = [
    (".lora_down", ".proj_down"),
    (".lora_up", ".proj_up"),
    (".smooth_orig", ".smooth_factor_orig"),
    (".smooth", ".smooth_factor"),
]


def _convert_checkpoint_key(k: str) -> str:
    """Rename nunchaku-converter keys to runtime parameter names."""
    for old, new in _KEY_RENAMES:
        if old in k:
            return k.replace(old, new)
    return k


def _convert_checkpoint_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    return {_convert_checkpoint_key(k): v for k, v in sd.items()}


def _infer_rank_from_converted_state_dict(sd: dict[str, Any]) -> int:
    for k, v in sd.items():
        if k.endswith(".proj_down") and getattr(v, "ndim", None) == 2:
            return int(v.shape[1])
    raise ValueError("Cannot infer SVD rank from checkpoint (missing any '*.proj_down' tensors).")


# ---------------------------------------------------------------------------
# SVDQ linear helper
# ---------------------------------------------------------------------------


def _make_svdq_linear(in_features: int, out_features: int, *, rank: int, precision: str, device, dtype):
    from nunchaku.models.linear import SVDQW4A4Linear

    return SVDQW4A4Linear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=True,
        precision=precision,
        torch_dtype=dtype,
        device=device,
    )


# ---------------------------------------------------------------------------
# Attention norms helper
# ---------------------------------------------------------------------------


def _build_attn_norms(*, head_dim: int, eps: float, with_added: bool, device, dtype) -> nn.ModuleDict:
    from diffusers.models.normalization import RMSNorm

    def _rms():
        return RMSNorm(head_dim, eps=eps, elementwise_affine=True).to(device=device, dtype=dtype)

    d: dict[str, nn.Module] = {"norm_q": _rms(), "norm_k": _rms()}
    if with_added:
        d["norm_added_q"] = _rms()
        d["norm_added_k"] = _rms()
    return nn.ModuleDict(d)


# ---------------------------------------------------------------------------
# Pure-Python QKV + Norm + Rotary helper
# ---------------------------------------------------------------------------


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding.

    Uses the same convention as diffusers' ``apply_rotary_emb`` from
    ``diffusers.models.embeddings``.

    Args:
        x: (B, S, H, D) query or key tensor.
        freqs_cis: (S, D/2, 2) or broadcastable rotary embedding (sin, cos stacked).
    """
    from diffusers.models.embeddings import apply_rotary_emb
    return apply_rotary_emb(x, freqs_cis, sequence_dim=1)


def _qkv_norm_rotary(hidden_states, qkv_proj, norm_q, norm_k, rotary_emb, heads: int):
    """Pure-Python replacement for fused_qkv_norm_rottary.

    Steps: QKV projection → chunk → RMSNorm(Q, K) → reshape to multi-head → rotary.
    """
    qkv = qkv_proj(hidden_states)
    query, key, value = qkv.chunk(3, dim=-1)

    # (B, S, dim) → (B, S, H, D)
    query = query.unflatten(-1, (heads, -1))
    key = key.unflatten(-1, (heads, -1))
    value = value.unflatten(-1, (heads, -1))

    query = norm_q(query)
    key = norm_k(key)

    if rotary_emb is not None:
        query = _apply_rotary_emb(query, rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)

    return query, key, value


# ---------------------------------------------------------------------------
# Attention dispatch
# ---------------------------------------------------------------------------


def _dispatch_attention(query, key, value, attention_mask):
    """Chroma attention dispatch (pure-Python path).

    ``query``/``key``/``value`` are (B, S, H, D).
    """
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    if attention_mask is None:
        return dispatch_attention_fn(query, key, value, attn_mask=None, backend=None)

    if attention_mask.ndim == 2 and query.shape[0] == 1:
        b, s = attention_mask.shape
        if b != 1:
            raise ValueError(f"Only batch_size=1 is supported for folded-mask fast path (got B={b}).")
        if int(query.shape[1]) != int(s) or int(key.shape[1]) != int(s):
            raise ValueError(
                f"Mask/sequence length mismatch: mask S={int(s)}, query S={int(query.shape[1])}, key S={int(key.shape[1])}"
            )

        m1 = attention_mask.to(dtype=query.dtype)[:, :, None, None].expand(
            query.shape[0], query.shape[1], query.shape[2], 1
        )
        d = int(query.shape[-1])
        scale = float(d) ** -0.5
        sqrt_d = float(d) ** 0.5
        extra = 8

        zeros_qk = m1.new_zeros((*m1.shape[:-1], extra - 1))
        q_ext = torch.cat([query, m1 * sqrt_d, zeros_qk], dim=-1)
        k_ext = torch.cat([key, m1, zeros_qk], dim=-1)
        v_ext = torch.cat([value, value.new_zeros((*value.shape[:-1], extra))], dim=-1)

        try:
            out_ext = dispatch_attention_fn(q_ext, k_ext, v_ext, attn_mask=None, backend="_native_flash", scale=scale)
        except TypeError:
            out_ext = None
        if out_ext is not None:
            return out_ext[..., :d]

    attn_mask_4d = _mask_to_4d(attention_mask)
    return dispatch_attention_fn(query, key, value, attn_mask=attn_mask_4d, backend="_native_efficient")


def _mask_to_4d(attention_mask):
    """Expand a 2D mask to a full QK mask (outer product), matching diffusers Chroma semantics."""
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        return attention_mask
    if attention_mask.ndim != 2:
        raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")
    return attention_mask[:, None, None, :] * attention_mask[:, None, :, None]


# ===========================================================================
# Transformer blocks
# ===========================================================================


class NunchakuChromaSingleTransformerBlock(nn.Module):
    def __init__(self, *, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float,
                 rank: int, precision: str, device, dtype, eps: float = 1e-6):
        super().__init__()
        from diffusers.models.transformers.transformer_chroma import ChromaAdaLayerNormZeroSinglePruned

        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = ChromaAdaLayerNormZeroSinglePruned(dim).to(device=device, dtype=dtype)
        self.attn = _build_attn_norms(head_dim=self.head_dim, eps=eps, with_added=False, device=device, dtype=dtype)

        lin = dict(rank=rank, precision=precision, device=device, dtype=dtype)
        self.qkv_proj = _make_svdq_linear(dim, 3 * dim, **lin)
        self.out_proj = _make_svdq_linear(dim, dim, **lin)
        self.mlp_fc1 = _make_svdq_linear(dim, mlp_hidden_dim, **lin)
        self.mlp_fc2 = _make_svdq_linear(mlp_hidden_dim, dim, **lin)

    def forward(self, hidden_states, temb, image_rotary_emb=None, attention_mask_1d=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        mlp_hidden = self.mlp_fc1(norm_hidden_states)
        mlp_hidden = F.gelu(mlp_hidden, approximate="tanh")
        mlp_out = self.mlp_fc2(mlp_hidden)

        query, key, value = _qkv_norm_rotary(
            norm_hidden_states, self.qkv_proj, self.attn["norm_q"], self.attn["norm_k"],
            image_rotary_emb, self.heads,
        )
        attn_out = _dispatch_attention(query, key, value, attention_mask_1d)
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        hidden_states = residual + gate.unsqueeze(1) * (self.out_proj(attn_out) + mlp_out)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class NunchakuChromaTransformerBlock(nn.Module):
    def __init__(self, *, dim: int, num_attention_heads: int, attention_head_dim: int,
                 rank: int, precision: str, device, dtype, eps: float = 1e-6):
        super().__init__()
        from diffusers.models.transformers.transformer_chroma import ChromaAdaLayerNormZeroPruned

        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

        self.norm1 = ChromaAdaLayerNormZeroPruned(dim).to(device=device, dtype=dtype)
        self.norm1_context = ChromaAdaLayerNormZeroPruned(dim).to(device=device, dtype=dtype)
        self.attn = _build_attn_norms(head_dim=self.head_dim, eps=eps, with_added=True, device=device, dtype=dtype)

        lin = dict(rank=rank, precision=precision, device=device, dtype=dtype)
        self.qkv_proj = _make_svdq_linear(dim, 3 * dim, **lin)
        self.qkv_proj_context = _make_svdq_linear(dim, 3 * dim, **lin)
        self.out_proj = _make_svdq_linear(dim, dim, **lin)
        self.out_proj_context = _make_svdq_linear(dim, dim, **lin)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype)

        self.mlp_fc1 = _make_svdq_linear(dim, 4 * dim, **lin)
        self.mlp_fc2 = _make_svdq_linear(4 * dim, dim, **lin)
        self.mlp_context_fc1 = _make_svdq_linear(dim, 4 * dim, **lin)
        self.mlp_context_fc2 = _make_svdq_linear(4 * dim, dim, **lin)
        self.mlp_context_fc2.act_unsigned = False

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                attention_mask_1d=None):
        temb_img, temb_txt = temb[:, :6], temb[:, 6:]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb_img)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb_txt
        )

        rotary_img, rotary_txt = image_rotary_emb
        txt_len = int(norm_encoder_hidden_states.shape[1])

        query, key, value = _qkv_norm_rotary(
            norm_hidden_states, self.qkv_proj, self.attn["norm_q"], self.attn["norm_k"], rotary_img, self.heads
        )
        c_query, c_key, c_value = _qkv_norm_rotary(
            norm_encoder_hidden_states, self.qkv_proj_context,
            self.attn["norm_added_q"], self.attn["norm_added_k"], rotary_txt, self.heads,
        )
        query = torch.cat([c_query, query], dim=1)
        key = torch.cat([c_key, key], dim=1)
        value = torch.cat([c_value, value], dim=1)

        attn_out = _dispatch_attention(query, key, value, attention_mask_1d)
        attn_out = attn_out.flatten(2, 3).to(query.dtype)
        context_attn_output, attn_output = attn_out.split_with_sizes([txt_len, attn_out.shape[1] - txt_len], dim=1)

        attn_output = self.out_proj(attn_output)
        context_attn_output = self.out_proj_context(context_attn_output)

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        nh = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        mlp_h = F.gelu(self.mlp_fc1(nh), approximate="tanh")
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp_fc2(mlp_h)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        ne = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        mlp_e = F.gelu(self.mlp_context_fc1(ne), approximate="tanh")
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.mlp_context_fc2(mlp_e)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


# ===========================================================================
# Top-level model
# ===========================================================================


class NunchakuChromaTransformer2dModel(ModelMixin, ConfigMixin):
    """Chroma transformer that loads the DeepCompressor/nunchaku-ext safetensors layout."""

    def __init__(self, *, config: dict[str, Any], rank: int, precision: str, device, dtype):
        super().__init__()
        from diffusers.models.transformers.transformer_chroma import (
            ChromaAdaLayerNormContinuousPruned,
            ChromaApproximator,
            ChromaCombinedTimestepTextProjEmbeddings,
        )
        from diffusers.models.transformers.transformer_flux import FluxPosEmbed

        patch_size = int(config["patch_size"])
        in_channels = int(config["in_channels"])
        out_channels = int(config.get("out_channels") or in_channels)
        num_layers = int(config["num_layers"])
        num_single_layers = int(config["num_single_layers"])
        attention_head_dim = int(config["attention_head_dim"])
        num_attention_heads = int(config["num_attention_heads"])
        joint_attention_dim = int(config["joint_attention_dim"])
        axes_dims_rope = tuple(config.get("axes_dims_rope", (16, 56, 56)))
        approx_channels = int(config.get("approximator_num_channels", 64))
        approx_hidden = int(config.get("approximator_hidden_dim", 5120))
        approx_layers = int(config.get("approximator_layers", 5))

        self.register_to_config(
            patch_size=patch_size, in_channels=in_channels, out_channels=out_channels,
            num_layers=num_layers, num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim, num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim, axes_dims_rope=axes_dims_rope,
            approximator_num_channels=approx_channels, approximator_hidden_dim=approx_hidden,
            approximator_layers=approx_layers,
        )
        self.nunchaku_precision = str(precision)
        self.nunchaku_rank = int(rank)

        self.out_channels = out_channels
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        def _to(m):
            return m.to(device=device, dtype=dtype)

        self.pos_embed = _to(FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope))
        self.time_text_embed = _to(ChromaCombinedTimestepTextProjEmbeddings(
            num_channels=approx_channels // 4,
            out_dim=3 * num_single_layers + 2 * 6 * num_layers + 2,
        ))
        self.distilled_guidance_layer = _to(ChromaApproximator(
            in_dim=approx_channels, out_dim=inner_dim, hidden_dim=approx_hidden, n_layers=approx_layers,
        ))
        self.context_embedder = _to(nn.Linear(joint_attention_dim, inner_dim, bias=True))
        self.x_embedder = _to(nn.Linear(in_channels, inner_dim, bias=True))

        block_kw = dict(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                        rank=rank, precision=precision, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            NunchakuChromaTransformerBlock(dim=inner_dim, **block_kw) for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            NunchakuChromaSingleTransformerBlock(dim=inner_dim, mlp_ratio=4.0, **block_kw) for _ in range(num_single_layers)
        ])

        self.norm_out = _to(ChromaAdaLayerNormContinuousPruned(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6))
        self.proj_out = _to(nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True))

        self.encoder_hid_proj = None
        self.offload = False
        self.transformer_block_offload_manager = None
        self.single_transformer_block_offload_manager = None

    # ---- offload management ------------------------------------------------

    def set_offload(self, offload: bool, **kwargs):
        from nunchaku.models.utils import CPUOffloadManager

        if offload == self.offload:
            return
        self.offload = offload
        if offload:
            use_pin_memory = kwargs.get("use_pin_memory", True)
            num_blocks_on_gpu = kwargs.get("num_blocks_on_gpu", 1)
            on_gpu = [self.pos_embed, self.time_text_embed, self.distilled_guidance_layer,
                      self.context_embedder, self.x_embedder, self.norm_out, self.proj_out]
            self.transformer_block_offload_manager = CPUOffloadManager(
                self.transformer_blocks, use_pin_memory=use_pin_memory,
                on_gpu_modules=on_gpu, num_blocks_on_gpu=num_blocks_on_gpu,
            )
            self.single_transformer_block_offload_manager = CPUOffloadManager(
                self.single_transformer_blocks, use_pin_memory=use_pin_memory,
                num_blocks_on_gpu=num_blocks_on_gpu,
            )
        else:
            self.transformer_block_offload_manager = None
            self.single_transformer_block_offload_manager = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def to(self, *args, **kwargs):
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present = "dtype" in kwargs or any(isinstance(arg, torch.dtype) for arg in args)
        for arg in args:
            if isinstance(arg, str):
                try:
                    torch.device(arg)
                    device_arg_or_kwarg_present = True
                except RuntimeError:
                    pass
        if dtype_present:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. Use the `torch_dtype` "
                "argument in `from_pretrained` instead."
            )
        if getattr(self, "offload", False) and device_arg_or_kwarg_present:
            warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
            return self
        return super(type(self), self).to(*args, **kwargs)

    def _iter_blocks(self, blocks, offload_manager):
        """Yield (index, block) with transparent offload management."""
        if offload_manager is not None:
            compute_stream = torch.cuda.current_stream()
            offload_manager.initialize(compute_stream)
            for i in range(len(blocks)):
                with torch.cuda.stream(compute_stream):
                    yield i, offload_manager.get_block(i)
                offload_manager.step(compute_stream)
        else:
            yield from enumerate(blocks)

    def _strip_accelerate_hooks_if_needed(self):
        """Neutralise accelerate per-module hooks that break quantized kernels."""
        try:
            from accelerate.hooks import remove_hook_from_module
        except ImportError:
            return

        children = [m for m in self.modules() if m is not self]
        has_hooks = any(getattr(m, "_hf_hook", None) is not None for m in children)
        if not has_hooks:
            return

        if not self._sequential_offload_warned:
            self._sequential_offload_warned = True
            warn(
                "Detected accelerate sequential hooks on sub-modules of "
                "NunchakuChromaTransformer2dModel.  These are incompatible with "
                "quantized CUDA kernels and will be removed.  "
                "Use transformer.set_offload(True) and "
                "pipe._exclude_from_cpu_offload.append('transformer') instead.",
                UserWarning,
            )

        for m in children:
            if getattr(m, "_hf_hook", None) is not None:
                remove_hook_from_module(m)

        top_hook = getattr(self, "_hf_hook", None)
        if top_hook is not None:
            top_hook.execution_device = None
            top_hook.offload = False
            if hasattr(top_hook, "io_same_device"):
                top_hook.io_same_device = False
            if hasattr(top_hook, "weights_map"):
                top_hook.weights_map = None

        if not self.offload:
            self.set_offload(True)

    # ---- loading -----------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *,
        device: str = "cuda", torch_dtype: Any = None, precision: str | None = None,
        rank: int | None = None, pin_memory: bool | str = "auto",
        offload: bool = False, verbose: bool = True, return_report: bool = False,
    ):
        from nunchaku.models.transformers.utils import patch_scale_key
        from nunchaku.utils import get_precision_from_quantization_config, load_state_dict_in_safetensors

        ckpt = Path(pretrained_model_name_or_path)
        if not ckpt.exists():
            raise FileNotFoundError(str(ckpt))

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        sd_raw, md = load_state_dict_in_safetensors(ckpt, device="cpu", return_metadata=True)
        for required_key in ("config", "quantization_config"):
            if required_key not in md:
                raise ValueError(f"Missing required safetensors metadata: {required_key!r}")

        config = json.loads(md["config"])
        if config.get("_class_name") != "ChromaTransformer2DModel":
            raise ValueError(f"Unexpected config._class_name={config.get('_class_name')!r}")

        inferred_precision = get_precision_from_quantization_config(json.loads(md["quantization_config"]))
        sd = _convert_checkpoint_state_dict(sd_raw)
        inferred_rank = _infer_rank_from_converted_state_dict(sd)

        if precision is not None and str(precision) != str(inferred_precision):
            raise ValueError(f"precision mismatch: {precision!r} vs checkpoint {inferred_precision!r}")
        if rank is not None and int(rank) != int(inferred_rank):
            raise ValueError(f"rank mismatch: {rank} vs checkpoint {inferred_rank}")

        model = cls(config=config, rank=inferred_rank, precision=str(inferred_precision),
                    device=torch.device(device), dtype=torch_dtype)

        patch_scale_key(model, sd)
        wanted = set(model.state_dict().keys())
        sd_filtered = {k: v for k, v in sd.items() if k in wanted}
        model.load_state_dict(sd_filtered, strict=True)

        if str(inferred_precision) == "int4":
            for block in model.transformer_blocks:
                for attr in ("qkv_proj", "qkv_proj_context", "mlp_context_fc2"):
                    layer = getattr(block, attr, None)
                    if layer is not None and hasattr(layer, "smooth_factor_orig"):
                        layer.smooth_factor.data.copy_(layer.smooth_factor_orig.data)

        if verbose:
            print("[nunchaku.chroma] loaded:", str(ckpt))

        model.set_offload(offload)

        if return_report:
            return model, LoadReport(config=config, precision=str(inferred_precision), rank=inferred_rank)
        return model

    # ---- forward -----------------------------------------------------------

    _sequential_offload_warned: bool = False

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None,
        img_ids=None, txt_ids=None, attention_mask=None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        controlnet_block_samples=None, controlnet_single_block_samples=None,
        return_dict: bool = True, controlnet_blocks_repeat: bool = False,
    ):
        del controlnet_blocks_repeat

        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
            raise NotImplementedError("ControlNet is not supported in NunchakuChromaTransformer2dModel")
        if joint_attention_kwargs:
            raise NotImplementedError("joint_attention_kwargs is not supported in NunchakuChromaTransformer2dModel")

        self._strip_accelerate_hooks_if_needed()

        if self.offload:
            device = hidden_states.device
            self.transformer_block_offload_manager.set_device(device)
            self.single_transformer_block_offload_manager.set_device(device)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        batch_size = int(hidden_states.shape[0])

        pooled_temb = self.distilled_guidance_layer(self.time_text_embed(timestep))
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        txt_tokens = int(encoder_hidden_states.shape[1])
        img_tokens = int(hidden_states.shape[1])
        attn_mask_1d = attention_mask

        freqs_cos, freqs_sin = image_rotary_emb
        rotary_emb_txt = (freqs_cos[:txt_tokens], freqs_sin[:txt_tokens])
        rotary_emb_img = (freqs_cos[txt_tokens:], freqs_sin[txt_tokens:])
        rotary_emb_single = image_rotary_emb

        num_layers = len(self.transformer_blocks)
        num_single = len(self.single_transformer_blocks)
        img_offset = 3 * num_single
        txt_offset = img_offset + 6 * num_layers

        dual_mgr = self.transformer_block_offload_manager if self.offload else None
        single_mgr = self.single_transformer_block_offload_manager if self.offload else None

        for i, block in self._iter_blocks(self.transformer_blocks, dual_mgr):
            img_mod = img_offset + 6 * i
            txt_mod = txt_offset + 6 * i
            temb = torch.cat(
                (pooled_temb[:, img_mod : img_mod + 6], pooled_temb[:, txt_mod : txt_mod + 6]), dim=1,
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt), attention_mask_1d=attn_mask_1d,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for i, block in self._iter_blocks(self.single_transformer_blocks, single_mgr):
            temb = pooled_temb[:, 3 * i : 3 * i + 3]
            hidden_states = block(
                hidden_states=hidden_states, temb=temb, image_rotary_emb=rotary_emb_single,
                attention_mask_1d=attn_mask_1d,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, pooled_temb[:, -2:])
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

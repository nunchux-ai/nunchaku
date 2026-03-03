"""
This module provides Nunchaku ZImageTransformer2DModel and its building blocks in Python.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import FeedForward as ZImageFeedForward
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel, ZImageTransformerBlock
from huggingface_hub import utils

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLFeedForward

from ...ops.gemm import svdq_gemm_w4a4_cuda
from ...ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ...utils import get_precision, pad_tensor
from ..attention import NunchakuBaseAttention
from ..attention_processors.zimage import NunchakuZSingleStreamAttnProcessor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key


class NunchakuZImageRopeHook:
    """
    Hook class for caching and substition of packed `freqs_cis` tensor.
    """

    def __init__(self):
        self.packed_cache = {}

    def __call__(self, module: nn.Module, input_args: tuple, input_kwargs: dict):
        freqs_cis: torch.Tensor = input_kwargs.get("freqs_cis", None)
        if freqs_cis is None:
            return None
        cache_key = freqs_cis.data_ptr()
        packed_freqs_cis = self.packed_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            packed_freqs_cis = torch.view_as_real(freqs_cis).unsqueeze(3)
            packed_freqs_cis = torch.flip(packed_freqs_cis, dims=[-1])
            packed_freqs_cis = pack_rotemb(pad_tensor(packed_freqs_cis, 256, 1))
            self.packed_cache[cache_key] = packed_freqs_cis
        new_input_kwargs = input_kwargs.copy()
        new_input_kwargs["freqs_cis"] = packed_freqs_cis
        return input_args, new_input_kwargs


class NunchakuZImageFusedModule(nn.Module):
    """
    Fused module for quantized QKV projection, RMS normalization, and rotary embedding for ZImage attention.

    Parameters
    ----------
    qkv : SVDQW4A4Linear
        Quantized QKV projection layer.
    norm_q : RMSNorm
        RMSNorm for query.
    norm_k : RMSNorm
        RMSNorm for key.
    """

    def __init__(self, qkv: SVDQW4A4Linear, norm_q: RMSNorm, norm_k: RMSNorm):
        super().__init__()
        for name, param in qkv.named_parameters(prefix="qkv_"):
            setattr(self, name.replace(".", ""), param)
        self.qkv_precision = qkv.precision
        self.qkv_out_features = qkv.out_features
        for name, param in norm_q.named_parameters(prefix="norm_q_"):
            setattr(self, name.replace(".", ""), param)
        for name, param in norm_k.named_parameters(prefix="norm_k_"):
            setattr(self, name.replace(".", ""), param)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        """
        Fuse QKV projection, RMS normalizaion and rotary embedding.

        Parameters
        ----------
        x : torch.Tensor
            The hidden states tensor
        freqs_cis : torch.Tensor, optional
            The rotary embedding tensor

        Returns
        -------
        The projection results of q, k, v. q result and k result are RMS-normalized and applied RoPE.
        """
        batch_size, seq_len, channels = x.shape
        x = x.view(batch_size * seq_len, channels)
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.qkv_proj_down,
            smooth=self.qkv_smooth_factor,
            fp4=self.qkv_precision == "nvfp4",
            pad_size=256,
        )
        output = torch.empty(batch_size * seq_len, self.qkv_out_features, dtype=x.dtype, device=x.device)
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qkv_qweight,
            out=output,
            ascales=ascales,
            wscales=self.qkv_wscales,
            lora_act_in=lora_act_out,
            lora_up=self.qkv_proj_up,
            bias=getattr(self, "qkv_bias", None),
            fp4=self.qkv_precision == "nvfp4",
            alpha=1.0 if self.qkv_precision == "nvfp4" else None,
            wcscales=self.qkv_wcscales if self.qkv_precision == "nvfp4" else None,
            norm_q=self.norm_q_weight,
            norm_k=self.norm_k_weight,
            rotary_emb=freqs_cis,
        )

        output = output.view(batch_size, seq_len, -1)
        return output


class NunchakuZImageAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized Attention module for ZImage with quantized and fused QKV projections.

    Parameters
    ----------
    other : Attention
        The original Attention module in ZImage model.
    processor : str, optional
        The attention processor to use ("flashattn2" or "nunchaku-fp16").
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuZImageAttention, self).__init__(processor)
        self.inner_dim = orig_attn.inner_dim
        self.query_dim = orig_attn.query_dim
        self.use_bias = orig_attn.use_bias
        self.dropout = orig_attn.dropout
        self.out_dim = orig_attn.out_dim
        self.context_pre_only = orig_attn.context_pre_only
        self.pre_only = orig_attn.pre_only
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor
        self.is_cross_attention = orig_attn.is_cross_attention

        # region sub-modules
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k
        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)
        # end of region

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for NunchakuZImageAttention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states for cross-attention.
        attention_mask : torch.Tensor, optional
            Attention mask.
        **cross_attention_kwargs
            Additional arguments for cross attention.

        Returns
        -------
        Output of the attention processor.
        """
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor ("flashattn2").

            - ``"flashattn2"``: Standard FlashAttention-2. See :class:`~nunchaku.models.attention_processors.zimage.NunchakuZSingleStreamAttnProcessor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuZSingleStreamAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


def _convert_z_image_ff(z_ff: ZImageFeedForward) -> FeedForward:
    """
    Replace custom FeedForward module in `ZImageTransformerBlock`s with standard FeedForward in diffusers lib.

    Parameters
    ----------
    z_ff : ZImageFeedForward
        The feed forward sub-module in the ZImageTransformerBlock module

    Returns
    -------
    FeedForward
        A diffusers FeedForward module which is equivalent to the input `z_ff`

    """
    assert isinstance(z_ff, ZImageFeedForward)
    assert z_ff.w1.in_features == z_ff.w3.in_features
    assert z_ff.w1.out_features == z_ff.w3.out_features
    assert z_ff.w1.out_features == z_ff.w2.in_features
    # NOTE:
    # During `from_pretrained`, the model is typically constructed on the `meta` device
    # and later materialized via `to_empty(...)` + `load_state_dict(...)`.
    # Creating a fresh diffusers `FeedForward` on CPU and then moving it to `meta`
    # would allocate large temporary tensors and can dominate load time.
    target_dtype = z_ff.w1.weight.dtype
    target_device = z_ff.w1.weight.device
    with torch.device(target_device):
        converted_ff = FeedForward(
            dim=z_ff.w1.in_features,
            dim_out=z_ff.w2.out_features,
            dropout=0.0,
            activation_fn="swiglu",
            inner_dim=z_ff.w2.in_features,
            bias=False,
        )
    converted_ff = converted_ff.to(dtype=target_dtype)
    return converted_ff


def replace_fused_module(module, incompatible_keys):
    assert isinstance(module, NunchakuZImageAttention)
    module.fused_module = NunchakuZImageFusedModule(module.to_qkv, module.norm_q, module.norm_k)
    del module.to_qkv
    del module.norm_q
    del module.norm_k


class NunchakuZImageFeedForward(NunchakuSDXLFeedForward):
    """
    Quantized feed-forward block for :class:`NunchakuZImageTransformerBlock`.

    Replaces linear layers in a FeedForward block with :class:`~nunchaku.models.linear.SVDQW4A4Linear` for quantized inference.

    Parameters
    ----------
    ff : FeedForward
        Source ZImage FeedForward module to quantize.
    **kwargs :
        Additional arguments for SVDQW4A4Linear.
    """

    def __init__(self, ff: ZImageFeedForward, **kwargs):
        converted_ff = _convert_z_image_ff(ff)
        # forward pass are equivalent to NunchakuSDXLFeedForward
        NunchakuSDXLFeedForward.__init__(self, converted_ff, **kwargs)


class NunchakuZImageTransformer2DModel(ZImageTransformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized ZImageTransformer2DModel.
    """

    def _patch_model(self, skip_refiners: bool = False, **kwargs):
        """
        Patch the model by replacing attention and feed_forward modules in the orginal ZImageTransformerBlock.

        Parameters
        ----------
        skip_refiners: bool
            Default to `False`
            if `True`, transformer blocks of `noise_refiner` and `context_refiner` will NOT be replaced.
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuZImageTransformer2DModel
            The patched model.
        """

        def _patch_transformer_block(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.attention = NunchakuZImageAttention(block.attention, **kwargs)
                block.attention.register_load_state_dict_post_hook(replace_fused_module)
                block.feed_forward = NunchakuZImageFeedForward(block.feed_forward, **kwargs)

        def _convert_feed_forward(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.feed_forward = _convert_z_image_ff(block.feed_forward)

        self.skip_refiners = skip_refiners
        _patch_transformer_block(self.layers)
        if skip_refiners:
            _convert_feed_forward(self.noise_refiner)
            _convert_feed_forward(self.context_refiner)
        else:
            _patch_transformer_block(self.noise_refiner)
            _patch_transformer_block(self.context_refiner)
        return self

    def register_rope_hook(self, rope_hook: NunchakuZImageRopeHook):
        self.rope_hook_handles = []
        for _, ly in enumerate(self.layers):
            self.rope_hook_handles.append(ly.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
        if not self.skip_refiners:
            for _, nr in enumerate(self.noise_refiner):
                self.rope_hook_handles.append(nr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
            for _, cr in enumerate(self.context_refiner):
                self.rope_hook_handles.append(cr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))

    def unregister_rope_hook(self):
        for h in self.rope_hook_handles:
            h.remove()
        self.rope_hook_handles.clear()

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        """
        Adapted from diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel#forward

        Register pre-forward hooks for caching and substitution of packed `freqs_cis` tensor for all attention submodules and unregister after forwarding is done.
        """
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return super().forward(x, t, cap_feats, patch_size, f_patch_size, return_dict)
        finally:
            self.unregister_rope_hook()
            del rope_hook

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuZImageTransformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuZImageTransformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for ZImageTransformer2DModel")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        rank = quantization_config.get("rank", 32)
        skip_refiners = quantization_config.get("skip_refiners", False)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        # print(f"quantization_config: {quantization_config}, rank={rank}, skip_refiners={skip_refiners}")

        transformer._patch_model(skip_refiners=skip_refiners, precision=precision, rank=rank, **kwargs)
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)

        transformer.load_state_dict(model_state_dict)

        return transformer

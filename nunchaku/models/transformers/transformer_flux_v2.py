"""
This module provides Nunchaku FluxTransformer2DModel and its building blocks in Python.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from huggingface_hub import utils
from torch import nn
from torch.nn import GELU

from ...lora.flux.nunchaku_converter import fuse_vectors, to_nunchaku
from ...lora.flux.utils import is_nunchaku_format
from ...ops.fused import fused_gelu_mlp
from ...utils import get_precision, load_state_dict_in_safetensors, pad_tensor
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.flux import NunchakuFluxFA2Processor, NunchakuFluxFP16AttnProcessor
from ..embeddings import NunchakuFluxPosEmbed, pack_rotemb
from ..linear import SVDQW4A4Linear
from ..normalization import NunchakuAdaLayerNormZero, NunchakuAdaLayerNormZeroSingle
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key


class NunchakuFluxAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized FluxAttention module with quantized and fused QKV projections.

    Parameters
    ----------
    other : FluxAttention
        The original FluxAttention module to wrap and quantize.
    processor : str, optional
        The attention processor to use ("flashattn2" or "nunchaku-fp16").
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: FluxAttention, processor: str = "flashattn2", **kwargs):
        super(NunchakuFluxAttention, self).__init__(processor)
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.out_dim = other.out_dim
        self.context_pre_only = other.context_pre_only
        self.pre_only = other.pre_only
        self.heads = other.heads
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k

        # Fuse the QKV projections for efficiency.
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        if not self.pre_only:
            self.to_out = other.to_out
            self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k

            # Fuse the additional QKV projections.
            with torch.device("meta"):
                add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
            self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, **kwargs)
            self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass for NunchakuFluxAttention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states for cross-attention.
        attention_mask : torch.Tensor, optional
            Attention mask.
        image_rotary_emb : tuple or torch.Tensor, optional
            Rotary embeddings for image/text tokens.
        **kwargs
            Additional arguments.

        Returns
        -------
        Output of the attention processor.
        """
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
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

            - ``"flashattn2"``: Standard FlashAttention-2. See :class:`~nunchaku.models.attention_processors.flux.NunchakuFluxFA2Processor`.
            - ``"nunchaku-fp16"``: Uses FP16 attention accumulation, up to 1.2× faster than FlashAttention-2 on NVIDIA 30-, 40-, and 50-series GPUs. See :class:`~nunchaku.models.attention_processors.flux.NunchakuFluxFP16AttnProcessor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuFluxFA2Processor()
        elif processor == "nunchaku-fp16":
            self.processor = NunchakuFluxFP16AttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuFluxTransformerBlock(FluxTransformerBlock):
    """
    Nunchaku-optimized FluxTransformerBlock with quantized attention and feedforward layers.

    Parameters
    ----------
    block : FluxTransformerBlock
        The original block to wrap and quantize.
    scale_shift : float, optional
        Value to add to scale parameters. Default is 1.0.
        Nunchaku may have already fused the scale_shift into the linear weights, so you may want to set it to 0.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: FluxTransformerBlock, scale_shift: float = 1, **kwargs):
        super(FluxTransformerBlock, self).__init__()
        self.scale_shift = scale_shift

        # The scale_shift=1 from AdaLayerNormZero has already been fused into the linear weights,
        # so we set scale_shift=0 here to avoid applying it again.
        self.norm1 = NunchakuAdaLayerNormZero(block.norm1, scale_shift=scale_shift)
        self.norm1_context = NunchakuAdaLayerNormZero(block.norm1_context, scale_shift=scale_shift)

        self.attn = NunchakuFluxAttention(block.attn, **kwargs)
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = NunchakuFeedForward(block.ff, **kwargs)
        self.ff_context = NunchakuFeedForward(block.ff_context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states for cross-attention.
        temb : torch.Tensor
            Time or conditioning embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary embeddings for image/text tokens.
        joint_attention_kwargs : dict, optional
            Additional attention arguments (not supported).

        Returns
        -------
        tuple
            (encoder_hidden_states, hidden_states) after block processing.

        Raises
        ------
        NotImplementedError
            If joint_attention_kwargs is provided.
        """
        if joint_attention_kwargs is not None and len(joint_attention_kwargs) > 0:
            raise NotImplementedError("joint_attention_kwargs is not supported")

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * scale_mlp[:, None] + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * c_scale_mlp[:, None] + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    """
    Nunchaku-optimized single transformer block with quantized attention and MLP.

    Parameters
    ----------
    block : FluxSingleTransformerBlock
        The original block to wrap and quantize.
    scale_shift : float, optional
        Value to add to scale parameters. Default is 1.0.
        Nunchaku may have already fused the scale_shift into the linear weights, so you may want to set it to 0.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: FluxSingleTransformerBlock, scale_shift: float = 1, **kwargs):
        super(FluxSingleTransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = block.norm
        self.norm = NunchakuAdaLayerNormZeroSingle(block.norm, scale_shift=scale_shift)

        self.mlp_fc1 = SVDQW4A4Linear.from_linear(block.proj_mlp, **kwargs)
        self.act_mlp = block.act_mlp
        self.mlp_fc2 = SVDQW4A4Linear.from_linear(block.proj_out, in_features=self.mlp_hidden_dim, **kwargs)
        # For int4, we shift the activation of mlp_fc2 to make it unsigned.
        self.mlp_fc2.act_unsigned = self.mlp_fc2.precision != "nvfp4"

        self.attn = NunchakuFluxAttention(block.attn, **kwargs)
        self.attn.to_out = SVDQW4A4Linear.from_linear(block.proj_out, in_features=self.mlp_fc1.in_features, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the single transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        temb : torch.Tensor
            Time or conditioning embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary embeddings for tokens.
        joint_attention_kwargs : dict, optional
            Additional attention arguments.

        Returns
        -------
        torch.Tensor
            Output hidden states after block processing.
        """
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        # Feedforward
        if isinstance(self.act_mlp, GELU):
            # Use fused GELU MLP for efficiency.
            mlp_hidden_states = fused_gelu_mlp(norm_hidden_states, self.mlp_fc1, self.mlp_fc2)
        else:
            # Fallback to original MLP.
            mlp_hidden_states = self.mlp_fc1(norm_hidden_states)
            mlp_hidden_states = self.act_mlp(mlp_hidden_states)
            mlp_hidden_states = self.mlp_fc2(mlp_hidden_states)

        # Attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs
        )

        hidden_states = attn_output + mlp_hidden_states
        gate = gate.unsqueeze(1)
        hidden_states = gate * hidden_states
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class NunchakuFluxTransformer2DModelV2(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized FluxTransformer2DModel.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super(NunchakuFluxTransformer2DModelV2, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        # these state_dicts are used for supporting lora
        self._unquantized_part_sd: dict[str, torch.Tensor] = {}
        self._unquantized_part_loras: dict[str, torch.Tensor] = {}
        self._quantized_part_sd: dict[str, torch.Tensor] = {}
        self._quantized_part_vectors: dict[str, torch.Tensor] = {}
        self._original_in_channels = self.x_embedder.in_features

    def _patch_model(self, **kwargs):
        """
        Patch the model with :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformerBlock`
        and :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxSingleTransformerBlock`.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuFluxTransformer2DModelV2
            The patched model.
        """
        self.pos_embed = NunchakuFluxPosEmbed(dim=self.inner_dim, theta=10000, axes_dim=self.pos_embed.axes_dim)
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFluxTransformerBlock(block, scale_shift=0, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFluxSingleTransformerBlock(block, scale_shift=0, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuFluxTransformer2DModelV2 from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuFluxTransformer2DModelV2
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
            raise NotImplementedError("Offload is not supported for FluxTransformer2DModelV2")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer: NunchakuFluxTransformer2DModelV2
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)

        quantized_part_sd: Dict[str, torch.Tensor] = {}
        unquantized_part_sd: Dict[str, torch.Tensor] = {}
        for k, v in model_state_dict.items():
            if k.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                quantized_part_sd[k] = v
            else:
                unquantized_part_sd[k] = v
        new_quantized_part_sd = {}
        for k, v in quantized_part_sd.items():
            if v.ndim == 1:
                new_quantized_part_sd[k] = v
            elif "qweight" in k:
                # only the shape information of this tensor is needed
                new_quantized_part_sd[k] = v.to("meta")

                # if the tensor has qweight, but does not have low-rank branch, we need to add some artificial tensors
                for t in ["lora_up", "lora_down"]:
                    new_k = k.replace(".qweight", f".{t}")
                    if new_k not in quantized_part_sd:
                        oc, ic = v.shape
                        ic = ic * 2  # v is packed into INT8, so we need to double the size
                        new_quantized_part_sd[k.replace(".qweight", f".{t}")] = torch.zeros(
                            (0, ic) if t == "lora_down" else (oc, 0), device=v.device, dtype=torch.bfloat16
                        )

            elif "lora" in k:
                new_quantized_part_sd[k] = v
        transformer._quantized_part_sd = new_quantized_part_sd
        transformer._unquantized_part_sd = unquantized_part_sd

        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision, rank=rank)

        transformer = transformer.to_empty(device=device)
        converted_state_dict = convert_flux_state_dict(model_state_dict)

        patch_scale_key(transformer, converted_state_dict)

        transformer.load_state_dict(converted_state_dict)

        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for the NunchakuFluxTransformer2DModelV2.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states of shape (batch_size, image_sequence_length, in_channels).
        encoder_hidden_states : torch.Tensor, optional
            Conditional embeddings (e.g., from text).
        pooled_projections : torch.Tensor, optional
            Projected embeddings from input conditions.
        timestep : torch.LongTensor, optional
            Denoising step.
        img_ids : torch.Tensor, optional
            Image token IDs.
        txt_ids : torch.Tensor, optional
            Text token IDs.
        guidance : torch.Tensor, optional
            Guidance tensor for classifier-free guidance.
        joint_attention_kwargs : dict, optional
            Additional attention arguments.
        controlnet_block_samples : any, optional
            Not supported.
        controlnet_single_block_samples : any, optional
            Not supported.
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).
        controlnet_blocks_repeat : bool, optional
            Not supported.

        Returns
        -------
        Transformer2DModelOutput or tuple
            Output sample tensor or output tuple.

        Raises
        ------
        NotImplementedError
            If controlnet is requested.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # Controlnet residual (not supported for now)
            if controlnet_block_samples is not None:
                raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # Controlnet residual (not supported for now)
            if controlnet_single_block_samples is not None:
                raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")

        hidden_states = hidden_states[:, txt_tokens:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def update_lora_params(self, path_or_state_dict: str | dict[str, torch.Tensor]):
        """
        Update the model with new LoRA parameters.

        Parameters
        ----------
        path_or_state_dict : str or dict
            Path to a LoRA weights file or a state dict. The path supports:

            - Local file path, e.g., ``"/path/to/your/lora.safetensors"``
            - HuggingFace repo with file, e.g., ``"user/repo/lora.safetensors"``
              (automatically downloaded and cached)
        """
        if isinstance(path_or_state_dict, dict):
            state_dict = {
                k: v for k, v in path_or_state_dict.items()
            }  # copy a new one to avoid modifying the original one
        else:
            state_dict = load_state_dict_in_safetensors(path_or_state_dict)

        if not is_nunchaku_format(state_dict):
            state_dict = to_nunchaku(state_dict, base_sd=self._quantized_part_sd)

        unquantized_part_loras = {}
        for k, v in list(state_dict.items()):
            device = next(self.parameters()).device
            if "transformer_blocks" not in k:
                unquantized_part_loras[k] = state_dict.pop(k).to(device)

        if len(self._unquantized_part_loras) > 0 or len(unquantized_part_loras) > 0:
            self._unquantized_part_loras = unquantized_part_loras

            self._unquantized_part_sd = {k: v for k, v in self._unquantized_part_sd.items() if "pulid_ca" not in k}
            unquantized_part_lora_state_dict = self._get_unquantized_part_lora_state_dict(1)
            state_dict.update(unquantized_part_lora_state_dict)

        quantized_part_vectors = {}
        for k, v in list(state_dict.items()):
            if v.ndim == 1:
                quantized_part_vectors[k] = state_dict.pop(k)
        if len(self._quantized_part_vectors) > 0 or len(quantized_part_vectors) > 0:
            self._quantized_part_vectors = quantized_part_vectors
            updated_vectors = fuse_vectors(quantized_part_vectors, self._quantized_part_sd, 1)
            state_dict.update(updated_vectors)

        state_dict = convert_flux_state_dict(state_dict)  # v1 to v2

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            logger = logging.getLogger(__name__)
            logger.debug(f"update_lora_params missing len={len(missing)}, unexpected len={len(unexpected)}")
            for i, unexpected_key in enumerate(unexpected):
                logger.debug(f"unexpected_key: {unexpected_key}")
                if i >= 10:
                    break

    def set_lora_strength(self, strength: float = 1):
        """
        Sets the LoRA scaling strength for the model.

        Note: This function can only be used with a single LoRA. For multiple LoRAs,
        please fuse the LoRA scale into the weights.

        Parameters
        ----------
        strength : float, optional
            LoRA scaling strength (default: 1).

        Note: This function will change the strength of all the LoRAs. So only use it when you only have a single LoRA.
        """
        for _, m in self.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                m.set_lora_scale(strength)
        new_state_dict = {}
        if len(self._unquantized_part_loras) > 0:
            unquantized_part_lora_state_dict = self._get_unquantized_part_lora_state_dict(strength)
            new_state_dict.update(unquantized_part_lora_state_dict)
        if len(self._quantized_part_vectors) > 0:
            vector_dict = fuse_vectors(self._quantized_part_vectors, self._quantized_part_sd, strength)
            new_state_dict.update(vector_dict)
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            logger = logging.getLogger(__name__)
            logger.debug(
                f"set_lora_strength state_dict missing key len={len(missing)}, unexpected key len={len(unexpected)}"
            )

    def _get_unquantized_part_lora_state_dict(self, strength: float = 1) -> dict[str, torch.Tensor]:
        """
        Updates the unquantized part of the model with LoRA parameters.

        Parameters
        ----------
        strength : float, optional
            LoRA scaling strength (default: 1).
        """
        # check if we need to expand the linear layers
        device = next(self.parameters()).device
        for k, v in self._unquantized_part_loras.items():
            if "lora_A" in k:
                lora_a = v
                lora_b = self._unquantized_part_loras[k.replace(".lora_A.", ".lora_B.")]
                diff_shape = (lora_b.shape[0], lora_a.shape[1])
                weight_shape = self._unquantized_part_sd[k.replace(".lora_A.", ".")].shape
                if diff_shape[0] > weight_shape[0] or diff_shape[1] > weight_shape[1]:
                    module_name = ".".join(k.split(".")[:-2])
                    self._expand_module(module_name, diff_shape)
            elif v.ndim == 1:
                diff_shape = v.shape
                weight_shape = self._unquantized_part_sd[k].shape
                if diff_shape[0] > weight_shape[0]:
                    assert diff_shape[0] >= weight_shape[0]
                    module_name = ".".join(k.split(".")[:-1])
                    module = self.get_submodule(module_name)
                    weight_shape = module.weight.shape
                    diff_shape = (diff_shape[0], weight_shape[1])
                    self._expand_module(module_name, diff_shape)
        new_state_dict = {}
        for k in self._unquantized_part_sd.keys():
            v = self._unquantized_part_sd[k]
            v = v.to(device)
            self._unquantized_part_sd[k] = v

            if v.ndim == 1 and k in self._unquantized_part_loras:
                diff = strength * self._unquantized_part_loras[k]
                if diff.shape[0] < v.shape[0]:
                    diff = torch.cat(
                        [diff, torch.zeros(v.shape[0] - diff.shape[0], device=device, dtype=v.dtype)], dim=0
                    )
                new_state_dict[k] = v + diff
            elif v.ndim == 2 and k.replace(".weight", ".lora_B.weight") in self._unquantized_part_loras:
                lora_a = self._unquantized_part_loras[k.replace(".weight", ".lora_A.weight")]
                lora_b = self._unquantized_part_loras[k.replace(".weight", ".lora_B.weight")]

                if lora_a.shape[1] < v.shape[1]:
                    lora_a = torch.cat(
                        [
                            lora_a,
                            torch.zeros(lora_a.shape[0], v.shape[1] - lora_a.shape[1], device=device, dtype=v.dtype),
                        ],
                        dim=1,
                    )
                if lora_b.shape[0] < v.shape[0]:
                    lora_b = torch.cat(
                        [
                            lora_b,
                            torch.zeros(v.shape[0] - lora_b.shape[0], lora_b.shape[1], device=device, dtype=v.dtype),
                        ],
                        dim=0,
                    )

                diff = strength * (lora_b @ lora_a)
                new_state_dict[k] = v + diff
            else:
                new_state_dict[k] = v
        return new_state_dict

    def _expand_module(self, module_name: str, new_shape: tuple[int, int]):
        """
        Expands a linear module to a new shape for LoRA compatibility.
        Mostly for FLUX.1-tools LoRA which changes the input channels.

        Parameters
        ----------
        module_name : str
            Name of the module to expand.
        new_shape : tuple[int, int]
            New shape (out_features, in_features) for the module.
        """
        module = self.get_submodule(module_name)
        assert isinstance(module, nn.Linear)
        weight_shape = module.weight.shape
        # logger.info("Expand the shape of module {} from {} to {}".format(module_name, tuple(weight_shape), new_shape))
        assert new_shape[0] >= weight_shape[0] and new_shape[1] >= weight_shape[1]
        new_module = nn.Linear(
            new_shape[1],
            new_shape[0],
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        new_module.weight.data.zero_()
        new_module.weight.data[: weight_shape[0], : weight_shape[1]] = module.weight.data
        self._unquantized_part_sd[f"{module_name}.weight"] = new_module.weight.data.clone()
        if new_module.bias is not None:
            new_module.bias.data.zero_()
            new_module.bias.data[: weight_shape[0]] = module.bias.data
            self._unquantized_part_sd[f"{module_name}.bias"] = new_module.bias.data.clone()
        parent_name = ".".join(module_name.split(".")[:-1])
        parent_module = self.get_submodule(parent_name)
        parent_module.add_module(module_name.split(".")[-1], new_module)

        if module_name == "x_embedder":
            new_value = int(new_module.weight.data.shape[1])
            old_value = getattr(self.config, "in_channels")
            if new_value != old_value:
                # logger.info(f"Update in_channels from {old_value} to {new_value}")
                setattr(self.config, "in_channels", new_value)


def convert_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from the :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`
    format to :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2` format.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The original state dict.

    Returns
    -------
    dict[str, torch.Tensor]
        The converted state dict compatible with :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2`.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if "single_transformer_blocks." in k:
            if ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        elif "transformer_blocks." in k:
            if ".mlp_context_fc1" in k:
                new_k = k.replace(".mlp_context_fc1.", ".ff_context.net.0.proj.")
            elif ".mlp_context_fc2" in k:
                new_k = k.replace(".mlp_context_fc2.", ".ff_context.net.2.")
            elif ".mlp_fc1" in k:
                new_k = k.replace(".mlp_fc1.", ".ff.net.0.proj.")
            elif ".mlp_fc2" in k:
                new_k = k.replace(".mlp_fc2.", ".ff.net.2.")
            elif ".qkv_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn.add_qkv_proj.")
            elif ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            elif ".norm_added_q." in k or ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_k = new_k.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.0.")
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn.to_add_out.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

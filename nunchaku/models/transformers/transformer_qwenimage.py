"""
This module provides implementations of NunchakuQwenImageTransformer2DModel and its building blocks.
"""

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import utils

from ...utils import get_precision
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.qwenimage import NunchakuQwenImageNaiveFA2Processor
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import CPUOffloadManager, fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key

logger = diffusers_logging.get_logger(__name__)


class NunchakuQwenAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized quantized attention module for QwenImage.

    Parameters
    ----------
    other : Attention
        The original QwenImage Attention module to wrap and quantize.
    processor : str, default="flashattn2"
        The attention processor to use.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuQwenAttention, self).__init__(processor)
        self.inner_dim = other.inner_dim
        self.inner_kv_dim = other.inner_kv_dim
        self.query_dim = other.query_dim
        self.use_bias = other.use_bias
        self.is_cross_attention = other.is_cross_attention
        self.cross_attention_dim = other.cross_attention_dim
        self.upcast_attention = other.upcast_attention
        self.upcast_softmax = other.upcast_softmax
        self.rescale_output_factor = other.rescale_output_factor
        self.residual_connection = other.residual_connection
        self.dropout = other.dropout
        self.fused_projections = other.fused_projections
        self.out_dim = other.out_dim
        self.out_context_dim = other.out_context_dim
        self.context_pre_only = other.context_pre_only
        self.pre_only = other.pre_only
        self.is_causal = other.is_causal
        self.scale_qk = other.scale_qk
        self.scale = other.scale
        self.heads = other.heads
        self.sliceable_head_dim = other.sliceable_head_dim
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.only_cross_attention = other.only_cross_attention
        self.group_norm = other.group_norm
        self.spatial_norm = other.spatial_norm

        self.norm_cross = other.norm_cross

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.norm_added_q = other.norm_added_q
        self.norm_added_k = other.norm_added_k

        # Fuse the QKV projections for quantization
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = other.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        assert self.added_kv_proj_dim is not None
        # Fuse the additional QKV projections
        with torch.device("meta"):
            add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
        self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, **kwargs)
        self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for NunchakuQwenAttention.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            Image stream input.
        encoder_hidden_states : torch.FloatTensor, optional
            Text stream input.
        encoder_hidden_states_mask : torch.FloatTensor, optional
            Mask for encoder hidden states.
        attention_mask : torch.FloatTensor, optional
            Attention mask.
        image_rotary_emb : torch.Tensor, optional
            Rotary embedding for images.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple
            Attention outputs for image and text streams.
        """
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor to use. Only "flashattn2" is supported for now. See :class:`~nunchaku.models.attention_processors.qwenimage.NunchakuQwenImageNaiveFA2Processor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuQwenImageNaiveFA2Processor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuQwenImageTransformerBlock(QwenImageTransformerBlock):
    """
    Quantized QwenImage Transformer Block.

    This block supports quantized linear layers and joint attention for image and text streams.

    Parameters
    ----------
    other : QwenImageTransformerBlock
        The original transformer block to wrap and quantize.
    scale_shift : float, default=1.0
        Value to add to scale parameters. Default is 1.0.
        Nunchaku may have already fused the scale_shift into the linear weights, so you may want to set it to 0.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: QwenImageTransformerBlock, scale_shift: float = 1.0, **kwargs):
        super(QwenImageTransformerBlock, self).__init__()

        self.dim = other.dim
        self.img_mod = other.img_mod
        self.img_mod[1] = AWQW4A16Linear.from_linear(other.img_mod[1], **kwargs)
        self.img_norm1 = other.img_norm1
        self.attn = NunchakuQwenAttention(other.attn, **kwargs)
        self.img_norm2 = other.img_norm2
        self.img_mlp = NunchakuFeedForward(other.img_mlp, **kwargs)

        # Text processing modules
        self.txt_mod = other.txt_mod
        self.txt_mod[1] = AWQW4A16Linear.from_linear(other.txt_mod[1], **kwargs)
        self.txt_norm1 = other.txt_norm1
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = other.txt_norm2
        self.txt_mlp = NunchakuFeedForward(other.txt_mlp, **kwargs)

        self.scale_shift = scale_shift

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply modulation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mod_params : torch.Tensor
            Modulation parameters.

        Returns
        -------
        tuple
            Modulated tensor and gate tensor.
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for NunchakuQwenImageTransformerBlock.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input.
        encoder_hidden_states : torch.Tensor
            Text stream input.
        encoder_hidden_states_mask : torch.Tensor
            Mask for encoder hidden states.
        temb : torch.Tensor
            Temporal embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary embedding for images.
        joint_attention_kwargs : dict, optional
            Additional arguments for joint attention.

        Returns
        -------
        tuple
            Updated encoder_hidden_states and hidden_states.
        """
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = (
            img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        )
        txt_mod_params = (
            txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
        )

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuQwenImageTransformer2DModel(QwenImageTransformer2DModel, NunchakuModelLoaderMixin):
    """
    Quantized QwenImage Transformer2DModel.

    This model supports quantized transformer blocks and optional CPU offloading for memory efficiency.

    Parameters
    ----------
    *args
        Positional arguments for the base model.
    **kwargs
        Keyword arguments for the base model and quantization.

    Attributes
    ----------
    offload : bool
        Whether CPU offloading is enabled.
    offload_manager : CPUOffloadManager or None
        Manager for offloading transformer blocks.
    _is_initialized : bool
        Whether the model has been patched for quantization.
    """

    def __init__(self, *args, **kwargs):
        self.offload = kwargs.pop("offload", False)
        self.offload_manager = None
        self._is_initialized = False
        super().__init__(*args, **kwargs)

    def _patch_model(self, **kwargs):
        """
        Patch the transformer blocks for quantization.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self
        """
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuQwenImageTransformerBlock(block, scale_shift=0, **kwargs)
        self._is_initialized = True
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a quantized model from a pretrained checkpoint.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the pretrained model checkpoint. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments for loading and quantization.

        Returns
        -------
        NunchakuQwenImageTransformer2DModel
            The loaded and quantized model.

        Raises
        ------
        AssertionError
            If the checkpoint is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata.get("config", "{}"))
        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision, rank=rank)

        transformer = transformer.to_empty(device=device)
        # need to re-init the pos_embed as to_empty does not work on it
        transformer.pos_embed = QwenEmbedRope(
            theta=10000, axes_dim=list(config.get("axes_dims_rope", [16, 56, 56])), scale_rope=True
        )

        patch_scale_key(transformer, model_state_dict)

        transformer.load_state_dict(model_state_dict)
        transformer.set_offload(offload)

        return transformer

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable asynchronous CPU offloading for transformer blocks.

        Parameters
        ----------
        offload : bool
            Whether to enable offloading.
        **kwargs
            Additional arguments for offload manager.

        See Also
        --------
        :class:`~nunchaku.models.utils.CPUOffloadManager`
        """
        if offload == self.offload:
            # nothing changed, just return
            return
        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                on_gpu_modules=[
                    self.img_in,
                    self.txt_in,
                    self.txt_norm,
                    self.time_text_embed,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for the Nunchaku QwenImage transformer model with ControlNet support.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input of shape `(batch_size, image_sequence_length, in_channels)`.
        encoder_hidden_states : torch.Tensor, optional
            Text stream input of shape `(batch_size, text_sequence_length, joint_attention_dim)`.
        encoder_hidden_states_mask : torch.Tensor, optional
            Mask for encoder hidden states of shape `(batch_size, text_sequence_length)`.
        timestep : torch.LongTensor, optional
            Timestep for temporal embedding.
        img_shapes : list of tuple, optional
            Image shapes for rotary embedding.
        txt_seq_lens : list of int, optional
            Text sequence lengths.
        guidance : torch.Tensor, optional
            Guidance tensor (for classifier-free guidance).
        attention_kwargs : dict, optional
            Additional attention arguments. A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
        controlnet_block_samples : optional
            ControlNet block samples for residual connections.
        return_dict : bool, default=True
            Whether to return a dict or tuple.

        Returns
        -------
        torch.Tensor or Transformer2DModelOutput
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        device = hidden_states.device
        if self.offload:
            self.offload_manager.set_device(device)

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        text_seq_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else None
        if text_seq_len is None:
            raise ValueError("encoder_hidden_states must be provided to compute text sequence length")
        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)
        for block_idx, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(block_idx)

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )

                # controlnet residual - same logic as in diffusers QwenImageTransformer2DModel
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

            if self.offload:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if self.offload:
            torch.cuda.empty_cache()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def to(self, *args, **kwargs):
        """
        Override the default ``.to()`` method.

        If offload is enabled, prevents moving the model to GPU.
        Prevents changing dtype after quantization.

        Parameters
        ----------
        *args
            Positional arguments for ``.to()``.
        **kwargs
            Keyword arguments for ``.to()``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If attempting to change dtype after quantization.
        """
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        # Try converting arguments to torch.device in case they are passed as strings
        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if dtype_present_in_args and self._is_initialized:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`."
            )
        if self.offload:
            if device_arg_or_kwarg_present:
                warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
                return self
        return super(type(self), self).to(*args, **kwargs)

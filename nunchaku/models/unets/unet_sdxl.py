"""
Implements the :class:`NunchakuSDXLUNet2DConditionModel`, providing Nunchaku quantized version of Stable Diffusion XL UNet2DConditionModel unet and its building blocks.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    Transformer2DModel,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from huggingface_hub import utils
from torch import nn

from nunchaku.utils import get_precision

from ..attention import NunchakuBaseAttention, _patch_linear
from ..attention_processors.sdxl import NunchakuSDXLFA2Processor
from ..linear import SVDQW4A4Linear
from ..transformers.utils import NunchakuModelLoaderMixin
from ..utils import fuse_linears


class NunchakuSDXLAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized SDXLAttention module with quantized and fused QKV projections.

    Parameters
    ----------
    orig_attn : Attention
        The original Attention module used by Stable Diffusion XL to wrap and quantize.
    processor : str, optional
        The attention processor to use (valid value: "flashattn2").
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuSDXLAttention, self).__init__(processor)

        self.is_cross_attention = orig_attn.is_cross_attention
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor

        if not orig_attn.is_cross_attention:
            # fuse the qkv
            with torch.device("meta"):
                to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
            self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        else:
            self.to_q = SVDQW4A4Linear.from_linear(orig_attn.to_q, **kwargs)
            self.to_k = orig_attn.to_k
            self.to_v = orig_attn.to_v

        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for NunchakuSDXLAttention.

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
            self,
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
            Name of the processor, "flashattn2" is supported. Others would be supported in future.

            - ``"flashattn2"``: Standard FlashAttention-2. Adapted from https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/models/attention_processor.py#AttnProcessor2_0

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuSDXLFA2Processor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuSDXLFeedForward(FeedForward):
    """
    Quantized feed-forward (MLP) block for :class:`NunchakuSDXLTransformerBlock`.

    Replaces linear layers in a FeedForward block with :class:`~nunchaku.models.linear.SVDQW4A4Linear` for quantized inference.

    Parameters
    ----------
    ff : FeedForward
        Source FeedForward block to quantize.
    **kwargs :
        Additional arguments for SVDQW4A4Linear.
    """

    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantized feed-forward block.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, D)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (B, D)
            Output tensor after feed-forward transformation.
        """
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class NunchakuSDXLTransformerBlock(BasicTransformerBlock):
    """
    Nunchaku-optimized transformer block for Stable Diffusion XL with quantized attention and feedforward layers.

    Parameters
    ----------
    block : BasicTransformerBlock
        The original block from within UNet2DConditionModel to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: BasicTransformerBlock, **kwargs):
        super(BasicTransformerBlock, self).__init__()

        self.norm_type = block.norm_type
        self.pos_embed = block.pos_embed
        self.only_cross_attention = block.only_cross_attention

        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.norm3 = block.norm3
        self.attn1 = NunchakuSDXLAttention(block.attn1, **kwargs)
        self.attn2 = NunchakuSDXLAttention(block.attn2, **kwargs)
        self.ff = NunchakuSDXLFeedForward(block.ff, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        attention_mask: torch.Tensor, optional
            The attention mask.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states for cross-attention.
        encoder_attention_mask: torch.Tensor, optional
            The encoder attention mask.
        cross_attention_kwargs: dict
            Addtional cross attention kwargs.


        Returns
        -------
        hidden_states: torch.Tensor
            The hidden states after processing.

        Raises
        ------
        ValueError
            If norm_type is not  "layer_norm" or only_cross_attetion is true.
        """

        # Adapted from diffusers.models.attention#BasicTransformerBlock#forward

        if self.norm_type == "layer_norm":
            norm_hidden_states = self.norm1(hidden_states)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if self.only_cross_attention:
            raise ValueError("only_cross_attetion cannot be True")
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control # TODO check
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:

            if self.norm_type == "layer_norm":
                norm_hidden_states = self.norm2(hidden_states)
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward

        norm_hidden_states = self.norm3(hidden_states)

        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class NunchakuSDXLShiftedConv2d(nn.Module):
    # Adapted from https://github.com/nunchux-ai/deepcompressor/blob/main/deepcompressor/nn/patch/conv.py#ShiftedConv2d
    def __init__(
        self,
        orig_in_channels,
        orig_out_channels,
        orig_kernel_size,
        orig_stride,
        orig_padding,
        orig_dilation,
        orig_groups,
        #  orig_bias,
        orig_padding_mode,
        orig_device,
        orig_dtype,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=orig_in_channels,
            out_channels=orig_out_channels,
            kernel_size=orig_kernel_size,
            stride=orig_stride,
            padding=orig_padding,
            dilation=orig_dilation,
            groups=orig_groups,
            bias=True,
            padding_mode=orig_padding_mode,
            device=orig_device,
            dtype=orig_dtype,
        )
        self.shift = nn.Parameter(torch.empty(1, 1, 1, 1, dtype=orig_dtype), requires_grad=False)  # hard code
        self.out_channels = orig_out_channels

        self.padding_size = self.conv._reversed_padding_repeated_twice
        if all(p == 0 for p in self.padding_size):
            self.padding_mode = ""
        elif orig_padding_mode == "zeros":
            self.padding_mode = "constant"
            # use shift
        else:
            self.padding_mode = orig_padding_mode
        self.conv.padding = "valid"
        self.conv.padding_mode = "zeros"
        self.conv._reversed_padding_repeated_twice = [0, 0] * len(self.conv.kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input + self.shift
        if self.padding_mode == "constant":
            input = F.pad(input, self.padding_size, mode=self.padding_mode, value=self.shift.item())
        elif self.padding_mode:
            input = F.pad(input, self.padding_size, mode=self.padding_mode, value=None)
        return self.conv(input)


class NunchakuSDXLConcatShiftedConv2d(nn.Module):
    # Adapted from https://github.com/nunchux-ai/deepcompressor/blob/main/deepcompressor/nn/patch/conv.py#ConcatConv2d
    def __init__(self, orig_conv: nn.Conv2d, split: int):
        super().__init__()
        splits = [split, orig_conv.in_channels - split] if orig_conv.in_channels - split > 0 else [split]
        assert len(splits) > 1, "ConcatShiftedConv2d requires at least 2 input channels"
        self.in_channels_list = splits
        self.in_channels = orig_conv.in_channels
        self.out_channels = orig_conv.out_channels
        self.convs = nn.ModuleList(
            [
                NunchakuSDXLShiftedConv2d(
                    split_in_channels,
                    orig_conv.out_channels,
                    orig_conv.kernel_size,
                    orig_conv.stride,
                    orig_conv.padding,
                    orig_conv.dilation,
                    orig_conv.groups,
                    # bias if idx == num_convs - 1 else False,
                    orig_conv.padding_mode,
                    orig_conv.weight.device,
                    orig_conv.weight.dtype,
                )
                for split_in_channels in splits
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # slice x based on in_channels_list
        x_splits: list[torch.Tensor] = x.split(self.in_channels_list, dim=1)
        # apply each conv to each slice (we have to make contiguous input for quantization)
        # out_splits = [conv(x_split.contiguous()) for conv, x_split in zip(self.convs, x_splits, strict=True)]
        out_splits = [conv(x_split) for conv, x_split in zip(self.convs, x_splits, strict=True)]
        # sum the results
        return sum(out_splits)


class NunchakuSDXLUNet2DConditionModel(UNet2DConditionModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized UNet2DConditionModel for Stable Diffusion XL.
    """

    def _patch_model(self, **kwargs):
        """
        Patch the model by replace the orginal BasicTransformerBlock with :class:`NunchakuSDXLTransformerBlock`

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuSDXLUNet2DConditionModel
            The patched model.
        """

        def _patch_attentions(block: CrossAttnDownBlock2D | CrossAttnUpBlock2D | UNetMidBlock2DCrossAttn):
            for _, attn in enumerate(block.attentions):
                assert isinstance(attn, Transformer2DModel), "Dual cross attention is not supported"
                nunchaku_sdxl_transformer_blocks = []
                for _, transformer_block in enumerate(attn.transformer_blocks):
                    assert isinstance(transformer_block, BasicTransformerBlock)
                    nunchaku_sdxl_transformer_block = NunchakuSDXLTransformerBlock(transformer_block, **kwargs)
                    nunchaku_sdxl_transformer_blocks.append(nunchaku_sdxl_transformer_block)
                attn.transformer_blocks = nn.ModuleList(nunchaku_sdxl_transformer_blocks)

        # _patch_resnets_convs is not used since the support from the inference engine is not completed.
        def _patch_resnets_convs(
            block: CrossAttnDownBlock2D | CrossAttnUpBlock2D | UNetMidBlock2DCrossAttn | UpBlock2D | DownBlock2D,
            up_block_idx: int | None = None,
        ):
            for resnet_idx, resnet in enumerate(block.resnets):
                if isinstance(block, (CrossAttnUpBlock2D, UpBlock2D)):
                    if resnet_idx == 0:
                        if up_block_idx == 0:
                            prev_block = self.mid_block
                        else:
                            prev_block = self.up_blocks[up_block_idx - 1]
                        split = prev_block.resnets[-1].conv2.out_channels
                    else:
                        split = block.resnets[resnet_idx - 1].conv2.out_channels
                    resnet.conv1 = NunchakuSDXLConcatShiftedConv2d(resnet.conv1, split)
                else:
                    resnet.conv1 = NunchakuSDXLShiftedConv2d(
                        resnet.conv1.in_channels,
                        resnet.conv1.out_channels,
                        resnet.conv1.kernel_size,
                        resnet.conv1.stride,
                        resnet.conv1.padding,
                        resnet.conv1.dilation,
                        resnet.conv1.groups,
                        #  orig_bias,
                        resnet.conv1.padding_mode,
                        resnet.conv1.weight.device,
                        resnet.conv1.weight.dtype,
                    )
                resnet.conv2 = NunchakuSDXLShiftedConv2d(
                    resnet.conv2.in_channels,
                    resnet.conv2.out_channels,
                    resnet.conv2.kernel_size,
                    resnet.conv2.stride,
                    resnet.conv2.padding,
                    resnet.conv2.dilation,
                    resnet.conv2.groups,
                    #  orig_bias,
                    resnet.conv2.padding_mode,
                    resnet.conv2.weight.device,
                    resnet.conv2.weight.dtype,
                )

        for _, down_block in enumerate(self.down_blocks):
            if isinstance(down_block, CrossAttnDownBlock2D):
                _patch_attentions(down_block)

        for _, up_block in enumerate(self.up_blocks):
            if isinstance(up_block, CrossAttnUpBlock2D):
                _patch_attentions(up_block)

        assert isinstance(self.mid_block, UNetMidBlock2DCrossAttn), "Only UNetMidBlock2DCrossAttn is supported"
        _patch_attentions(self.mid_block)

        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuSDXLUNet2DConditionModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuSDXLUNet2DConditionModel
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
            raise NotImplementedError("Offload is not supported for NunchakuSDXLUNet2DConditionModel")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_path, str):
            pretrained_model_path = Path(pretrained_model_path)

        assert pretrained_model_path.is_file() or pretrained_model_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"

        unet, model_state_dict, metadata = cls._build_model(pretrained_model_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)
        unet = unet.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        unet._patch_model(precision=precision, rank=rank)
        unet = unet.to_empty(device=device)
        converted_state_dict = convert_sdxl_state_dict(model_state_dict)

        unet.load_state_dict(converted_state_dict)

        return unet


def convert_sdxl_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        if ".transformer_blocks." in k:
            if ".lora_down" in k:
                new_k = k.replace(".lora_down", ".proj_down")
            elif ".lora_up" in k:
                new_k = k.replace(".lora_up", ".proj_up")
            elif ".smooth_orig" in k:
                new_k = k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = k.replace(".smooth", ".smooth_factor")
            else:
                new_k = k
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

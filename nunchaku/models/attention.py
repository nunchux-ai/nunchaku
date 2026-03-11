"""
Nunchaku quantized attention-related modules.
"""

import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from torch import nn

from ..ops.fused import fused_gelu_mlp
from .linear import SVDQW4A4Linear


class NunchakuBaseAttention(nn.Module):
    """
    Base class for Nunchaku attention modules.

    Provides a common interface for attention modules with processor selection.

    Parameters
    ----------
    processor : str, optional
        Name of the attention processor to use. Default is "flashattn2".
    *args, **kwargs :
        Additional arguments for subclass initialization.
    """

    def __init__(self, processor: str = "flashattn2", *args, **kwargs):
        super(NunchakuBaseAttention, self).__init__()
        self.processor = None
        self.set_processor(processor)

    def set_processor(self, processor: str):
        """
        Set the attention processor. Must be implemented by subclasses.

        Parameters
        ----------
        processor : str
            Name of the processor to use.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        raise NotImplementedError("Subclass must implement this method")


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    """
    Recursively replace all nn.Linear modules in a given module with a custom linear class.

    Parameters
    ----------
    module : nn.Module
        The module to patch.
    linear_cls : type
        The custom linear class to use for replacement.
    **kwargs :
        Additional arguments passed to ``from_linear``.

    Returns
    -------
    nn.Module
        The patched module with custom linear layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuFeedForward(FeedForward):
    """
    Quantized feed-forward (MLP) block with fused GELU support.

    Replaces linear layers in a FeedForward block with :class:`~nunchaku.models.linear.SVDQW4A4Linear` for quantized inference.
    Supports fused GELU-MLP computation for efficiency.

    Parameters
    ----------
    ff : FeedForward
        Source FeedForward block to quantize.
    **kwargs :
        Additional arguments for SVDQW4A4Linear.

    Notes
    -----
    For int4 quantization, the activation of the second MLP layer is shifted to be unsigned.
    """

    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **kwargs)
        # For int4, shift the activation of mlp_fc2 to make it unsigned
        self.net[2].act_unsigned = self.net[2].precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantized feed-forward block.
        It will call :func:`~nunchaku.ops.fused.fused_gelu_mlp` if the first layer is GELU;
        otherwise, apply modules sequentially.

        Parameters
        ----------
        hidden_states : torch.Tensor, shape (B, D)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (B, D)
            Output tensor after feed-forward transformation.
        """
        if isinstance(self.net[0], GELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            # Fallback to original implementation
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states

    def update_lora_params(self, lora_dict: dict[str, torch.Tensor]):
        """
        Update LoRA parameters for the feed-forward network.

        This method handles LoRA weights for the MLP layers in the feed-forward network.
        The LoRA weights are in Nunchaku format (packed lora_down/lora_up) and are
        directly replaced into the low-rank projections.

        Parameters
        ----------
        lora_dict : dict[str, torch.Tensor]
            Dictionary containing LoRA weights for this feed-forward module.
            Expected keys: 'net.0.proj.lora_down', 'net.0.proj.lora_up', 'net.2.lora_down', 'net.2.lora_up'
        """
        import logging

        from ..linear import SVDQW4A4Linear

        logger = logging.getLogger(__name__)

        # Helper function to apply LoRA to a SVDQW4A4Linear layer
        def apply_lora_to_linear(linear_layer, lora_dict, layer_prefix):
            lora_down_key = None
            lora_up_key = None

            # Find lora_down and lora_up for this layer
            for k in lora_dict.keys():
                if layer_prefix in k:
                    if "lora_down" in k:
                        lora_down_key = k
                    elif "lora_up" in k:
                        lora_up_key = k

            if lora_down_key is None or lora_up_key is None:
                return  # No LoRA for this layer

            lora_down_packed = lora_dict[lora_down_key]
            lora_up_packed = lora_dict[lora_up_key]

            device = linear_layer.proj_down.device
            dtype = linear_layer.proj_down.dtype

            # The LoRA weights are already merged with original low-rank branches in the converter
            # Just directly apply them
            old_rank = linear_layer.rank

            linear_layer.proj_down.data = lora_down_packed.to(device=device, dtype=dtype)
            linear_layer.proj_up.data = lora_up_packed.to(device=device, dtype=dtype)

            # Update rank based on the merged weights
            new_rank = lora_down_packed.shape[1]
            linear_layer.rank = new_rank

            logger.debug(
                f"  ✅ Applied LoRA to {layer_prefix}: rank {old_rank} → {new_rank}, "
                f"proj_down shape={linear_layer.proj_down.shape}, "
                f"proj_up shape={linear_layer.proj_up.shape}"
            )

        # Apply LoRA to each SVDQW4A4Linear layer in the network
        for i, module in enumerate(self.net):
            if isinstance(module, SVDQW4A4Linear):
                apply_lora_to_linear(module, lora_dict, f"net.{i}")
            elif isinstance(module, GELU) and hasattr(module, "proj") and isinstance(module.proj, SVDQW4A4Linear):
                # For GELU with proj attribute
                apply_lora_to_linear(module.proj, lora_dict, f"net.{i}.proj")

    def restore_original_params(self):
        """
        Note: For Qwen Image, LoRA removal is handled by reloading the model.
        There's no need to manually restore parameters since the converter
        merges LoRA with the original low-rank branches.
        """
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("  ℹ️  LoRA removal: reload model to restore original state")

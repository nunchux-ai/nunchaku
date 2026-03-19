"""
Utility functions for LoRAs in Flux models.
"""

import typing as tp

import torch

from ...utils import ceil_divide, load_state_dict_in_safetensors


def is_nunchaku_format(lora: str | dict[str, torch.Tensor]) -> bool:
    """
    Check if LoRA weights are in Nunchaku format.

    Parameters
    ----------
    lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a dictionary of LoRA weights.

    Returns
    -------
    bool
        True if the weights are in Nunchaku format, False otherwise.

    Examples
    --------
    >>> is_nunchaku_format("path/to/lora.safetensors")
    True
    """
    if isinstance(lora, str):
        tensors = load_state_dict_in_safetensors(lora, device="cpu", return_metadata=False)
        assert isinstance(tensors, dict), "Expected dict when return_metadata=False"
    else:
        tensors = lora

    for k in tensors.keys():
        if ".mlp_fc" in k or "mlp_context_fc1" in k:
            return True
    return False


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    """
    Pad a tensor so specified dimensions are divisible by given divisors.

    Parameters
    ----------
    tensor : torch.Tensor or None
        The tensor to pad. If None, returns None.
    divisor : int or sequence of int
        Divisor(s) for the dimension(s) to pad.
    dim : int or sequence of int
        Dimension(s) to pad.
    fill_value : float or int, optional
        Value to use for padding (default: 0).

    Returns
    -------
    torch.Tensor or None
        The padded tensor, or None if input tensor was None.

    Examples
    --------
    >>> tensor = torch.randn(10, 20)
    >>> pad(tensor, divisor=16, dim=0).shape
    torch.Size([16, 20])
    >>> pad(tensor, divisor=[16, 32], dim=[0, 1]).shape
    torch.Size([16, 32])
    """
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    indices = tuple(slice(0, extent) for extent in tensor.shape)
    result[indices] = tensor
    return result

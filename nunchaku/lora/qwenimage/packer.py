"""
Weight packing utilities for Qwen Image LoRA in Nunchaku quantization.

This module provides tools for packing Qwen Image LoRA weight tensors
for efficient GPU computation using Nunchaku's quantization infrastructure.
"""

import torch

from .utils import pad


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """
    Pack the low-rank weight tensor for W4A4 linear layers in Qwen Image.

    Parameters
    ----------
    weight : torch.Tensor
        Low-rank weight tensor.
    down : bool
        If True, pack as down-projection; else as up-projection.

    Returns
    -------
    torch.Tensor
        Packed weight tensor.
    """
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    lane_n, lane_k = 1, 2  # lane_n is always 1, lane_k is 32 bits // 16 bits = 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = pad(weight, divisor=(frag_n, frag_k), dim=(0, 1))
    if down:
        r, c = weight.shape
        r_frags, c_frags = r // frag_n, c // frag_k
        weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
    else:
        c, r = weight.shape
        c_frags, r_frags = c // frag_n, r // frag_k
        weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)
    weight = weight.reshape(c_frags, r_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return weight.view(c, r)


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """
    Unpack the low-rank weight tensor from W4A4 linear layers in Qwen Image.

    Parameters
    ----------
    weight : torch.Tensor
        Packed low-rank weight tensor.
    down : bool
        If True, unpack as down-projection; else as up-projection.

    Returns
    -------
    torch.Tensor
        Unpacked weight tensor.
    """
    c, r = weight.shape
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    lane_n, lane_k = 1, 2  # lane_n is always 1, lane_k is 32 bits // 16 bits = 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    if down:
        r_frags, c_frags = r // frag_n, c // frag_k
    else:
        c_frags, r_frags = c // frag_n, r // frag_k
    weight = weight.view(c_frags, r_frags, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, lane_k)
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(c_frags, r_frags, frag_n, frag_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight

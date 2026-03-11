"""
Nunchaku LoRA format converter for Qwen Image models.

This module provides utilities to convert LoRA weights from Diffusers format
to Nunchaku format for efficient quantized inference in Qwen Image models.

Key functions
-------------
- :func:`to_nunchaku` : Main conversion entry point
- :func:`convert_to_nunchaku_qwenimage_lowrank_dict` : Convert full model LoRA weights
"""

import logging
import os

import torch
from safetensors.torch import save_file

from ...utils import filter_state_dict, load_state_dict_in_safetensors
from .diffusers_converter import to_diffusers
from .packer import pack_lowrank_weight, unpack_lowrank_weight
from .utils import is_diffusers_format, is_nunchaku_format, pad

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# region utilities


def update_state_dict(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, torch.Tensor]:
    """
    Update a state dictionary with values from another, optionally adding a prefix to keys.

    Parameters
    ----------
    lhs : dict[str, torch.Tensor]
        Target state dictionary.
    rhs : dict[str, torch.Tensor]
        Source state dictionary.
    prefix : str, optional
        Prefix to add to keys from rhs.

    Returns
    -------
    dict[str, torch.Tensor]
        Updated state dictionary.

    Raises
    ------
    AssertionError
        If any key already exists in the target dictionary.
    """
    for rkey, value in rhs.items():
        lkey = f"{prefix}.{rkey}" if prefix else rkey
        assert lkey not in lhs, f"Key {lkey} already exists in the state dict."
        lhs[lkey] = value
    return lhs


# endregion


def reorder_adanorm_lora_up(lora_up: torch.Tensor, splits: int) -> torch.Tensor:
    """
    Reorder AdaNorm LoRA up-projection tensor for correct shape.

    Parameters
    ----------
    lora_up : torch.Tensor
        LoRA up-projection tensor.
    splits : int
        Number of splits for AdaNorm.

    Returns
    -------
    torch.Tensor
        Reordered tensor.
    """
    c, r = lora_up.shape
    assert c % splits == 0
    return lora_up.view(splits, c // splits, r).transpose(0, 1).reshape(c, r).contiguous()


def convert_to_nunchaku_transformer_block_lowrank_dict(
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    default_dtype: torch.dtype = torch.bfloat16,
    skip_base_merge: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights for a Qwen Image transformer block from Diffusers to Nunchaku format.

    This function handles the complex weight conversion and packing required for Qwen Image's
    dual-stream architecture with AWQW4A16Linear and SVDQW4A4Linear quantization.

    Parameters
    ----------
    orig_state_dict : dict[str, torch.Tensor]
        Original state dict with base model LoRA weights.
    extra_lora_dict : dict[str, torch.Tensor]
        Extra LoRA weights to merge.
    converted_block_name : str
        Block name for output (e.g., "transformer_blocks.0").
    candidate_block_name : str
        Block name for input lookup (e.g., "transformer.blocks.0").
    default_dtype : torch.dtype, optional
        Output tensor dtype (default: torch.bfloat16).
    skip_base_merge : bool, optional
        If True, skip merging with base model low-rank branches.
        Used for composed LoRAs that are already concatenated (default: False).

    Returns
    -------
    dict[str, torch.Tensor]
        Converted LoRA weights in Nunchaku format.
    """
    converted_dict = {}

    # Define module mapping for Qwen Image dual-stream architecture
    # Each entry maps (output_name, [input_candidates], conversion_type)
    modules_to_convert = {
        # Image stream modules
        "attn.to_qkv": (["attn.to_qkv"], "linear"),  # Fused QKV for image stream
        "attn.to_out.0": (["attn.to_out.0"], "linear"),  # Output projection for image stream
        "attn.add_qkv_proj": (["attn.add_qkv_proj"], "linear"),  # Fused QKV for text stream
        "attn.to_add_out": (["attn.to_add_out"], "linear"),  # Output projection for text stream
        "img_mod.1": (["img_mod.1"], "adanorm_zero"),  # Image modulation (6 splits)
        "img_mlp.net.0.proj": (["img_mlp.net.0.proj"], "linear"),  # Image MLP first projection
        "img_mlp.net.2": (["img_mlp.net.2"], "linear"),  # Image MLP second projection
        # Text stream modules
        "txt_mod.1": (["txt_mod.1"], "adanorm_zero"),  # Text modulation (6 splits)
        "txt_mlp.net.0.proj": (["txt_mlp.net.0.proj"], "linear"),  # Text MLP first projection
        "txt_mlp.net.2": (["txt_mlp.net.2"], "linear"),  # Text MLP second projection
    }

    for local_name, (candidate_names, convert_type) in modules_to_convert.items():
        orig_proj_down = None
        orig_proj_up = None

        # Try to find original proj_down/proj_up in base model state dict
        for candidate_name in candidate_names:
            # Look for proj_down and proj_up (original low-rank branches)
            orig_key_down = f"{converted_block_name}.{candidate_name}.proj_down"
            orig_key_up = f"{converted_block_name}.{candidate_name}.proj_up"
            if orig_key_down in orig_state_dict:
                orig_proj_down = orig_state_dict[orig_key_down]
                orig_proj_up = orig_state_dict[orig_key_up]
                break

        # Try to find in extra LoRA dict
        extra_lora_down = None
        extra_lora_up = None
        for candidate_name in candidate_names:
            extra_key_down = f"{candidate_block_name}.{candidate_name}.lora_A.weight"
            extra_key_up = f"{candidate_block_name}.{candidate_name}.lora_B.weight"
            if extra_key_down in extra_lora_dict:
                # Keep lora_A in its original format to match unpacked original LoRA
                # lora_A: [rank, in_features] (same as unpacked lora_down)
                # lora_B: [out_features, rank] (same as unpacked lora_up)
                extra_lora_down = extra_lora_dict[extra_key_down]
                extra_lora_up = extra_lora_dict[extra_key_up]
                break

        # Merge original low-rank branches with LoRA following Flux's approach
        if skip_base_merge:
            # For composed LoRAs, don't merge with base - they're already concatenated
            if extra_lora_down is not None:
                lora_down = extra_lora_down
                lora_up = extra_lora_up
                if local_name == "attn.to_qkv":  # Only log first layer to reduce noise
                    logger.info(
                        f"  [SKIP_MERGE] {converted_block_name}.{local_name}: lora_down={lora_down.shape}, lora_up={lora_up.shape}"
                    )
            else:
                continue  # No LoRA for this module
        elif orig_proj_down is not None and extra_lora_down is not None:
            # Unpack original proj_down/proj_up (these are the model's original low-rank branches)
            orig_proj_down_unpacked = unpack_lowrank_weight(orig_proj_down, down=True)
            orig_proj_up_unpacked = unpack_lowrank_weight(orig_proj_up, down=False)

            # Check if original is empty (like Flux's numel() == 0 check)
            if orig_proj_down_unpacked.numel() == 0 or orig_proj_up_unpacked.numel() == 0:
                # Empty base, use LoRA directly
                lora_down = extra_lora_down
                lora_up = extra_lora_up
            else:
                # Merge with base
                lora_down = torch.cat([orig_proj_down_unpacked, extra_lora_down], dim=0)
                lora_up = torch.cat([orig_proj_up_unpacked, extra_lora_up], dim=1)
        elif orig_proj_down is not None:
            lora_down = unpack_lowrank_weight(orig_proj_down, down=True)
            lora_up = unpack_lowrank_weight(orig_proj_up, down=False)
        elif extra_lora_down is not None:
            lora_down = extra_lora_down
            lora_up = extra_lora_up
        else:
            continue  # No LoRA for this module

        # Convert to target dtype
        lora_down = lora_down.to(default_dtype)
        lora_up = lora_up.to(default_dtype)

        if convert_type == "adanorm_zero":
            # AdaNorm with 6 splits (scale and shift for 3 parameters)
            lora_down_packed = pad(lora_down, divisor=16, dim=0)
            lora_up_packed = pad(reorder_adanorm_lora_up(lora_up, splits=6), divisor=16, dim=1)
        elif convert_type == "adanorm_single":
            # AdaNorm with 3 splits (single parameter)
            lora_down_packed = pad(lora_down, divisor=16, dim=0)
            lora_up_packed = pad(reorder_adanorm_lora_up(lora_up, splits=3), divisor=16, dim=1)
        elif convert_type == "linear":
            # Standard linear layer packing
            lora_down_packed = pack_lowrank_weight(lora_down, down=True)
            lora_up_packed = pack_lowrank_weight(lora_up, down=False)
        else:
            raise ValueError(f"Unknown conversion type: {convert_type}")
        # Store in converted dict
        converted_dict[f"{converted_block_name}.{local_name}.lora_down"] = lora_down_packed
        converted_dict[f"{converted_block_name}.{local_name}.lora_up"] = lora_up_packed

    return converted_dict


def fuse_vectors(
    vectors: dict[str, torch.Tensor], base_sd: dict[str, torch.Tensor], strength: float = 1
) -> dict[str, torch.Tensor]:
    """
    Fuse vector (bias) terms from LoRA into the base model for Qwen Image.

    This function handles the fusion of LoRA bias terms into the quantized base model,
    which is essential for maintaining model accuracy with quantized weights.

    Parameters
    ----------
    vectors : dict[str, torch.Tensor]
        LoRA vector terms (bias, normalization parameters).
    base_sd : dict[str, torch.Tensor]
        Base model state dict.
    strength : float, optional
        Scaling factor for LoRA vectors.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict with fused vectors.

    Notes
    -----
    - Handles dual-stream architecture (img_* and txt_* modules)
    - Supports AWQW4A16Linear and SVDQW4A4Linear quantization
    - Processes modulation parameters (img_mod, txt_mod)
    - Handles normalization layers (norm_q, norm_k, norm_added_q, norm_added_k)
    """
    tensors: dict[str, torch.Tensor] = {}

    for k, v in base_sd.items():
        if v.ndim != 1 or "smooth" in k:
            continue

        # Handle normalization layers (don't apply LoRA to these)
        if "norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k:
            tensors[k] = v
            continue

        # Handle modulation parameters (img_mod, txt_mod)
        if "mod." in k and ("img_mod" in k or "txt_mod" in k):
            # Modulation parameters need special handling for dual-stream architecture
            # Look for corresponding LoRA vectors
            lora_key = k.replace("mod.", "mod.1.")  # Map to the linear layer
            diff = vectors.get(lora_key, None)
            if diff is not None:
                # Apply LoRA to modulation parameters
                diff = diff * strength
                # Pad to match quantization requirements
                diff = pad(diff, divisor=16, dim=0)
                tensors[k] = v + diff
            else:
                tensors[k] = v
        else:
            # Handle other linear layers
            if k.startswith("transformer_blocks."):
                # Map Qwen Image specific layer names
                name_map = {
                    ".attn.to_qkv.": ".attn.to_qkv.",
                    ".attn.add_qkv_proj.": ".attn.add_qkv_proj.",
                    ".attn.to_out.0.": ".attn.to_out.0.",
                    ".attn.to_add_out.": ".attn.to_add_out.",
                    ".img_mlp.net.0.proj.": ".img_mlp.net.0.proj.",
                    ".img_mlp.net.2.": ".img_mlp.net.2.",
                    ".txt_mlp.net.0.proj.": ".txt_mlp.net.0.proj.",
                    ".txt_mlp.net.2.": ".txt_mlp.net.2.",
                }

                for original_pattern, new_pattern in name_map.items():
                    if original_pattern in k:
                        new_k = k.replace(original_pattern, new_pattern)
                        diff = vectors.get(new_k, None)
                        if diff is not None:
                            diff = diff * strength
                            # Apply padding for quantization compatibility
                            diff = pad(diff, divisor=16, dim=0)
                            tensors[k] = v + diff
                            break
                else:
                    tensors[k] = v
            else:
                tensors[k] = v

    return tensors


def convert_to_nunchaku_qwenimage_lowrank_dict(
    base_model: dict[str, torch.Tensor] | str,
    lora: dict[str, torch.Tensor] | str,
    default_dtype: torch.dtype = torch.bfloat16,
    skip_base_merge: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a base model and LoRA weights from Diffusers format to Nunchaku format for Qwen Image.

    Parameters
    ----------
    base_model : dict[str, torch.Tensor] or str
        Base model weights or path to safetensors file.
    lora : dict[str, torch.Tensor] or str
        LoRA weights or path to safetensors file.
    default_dtype : torch.dtype, optional
        Output tensor dtype (default: torch.bfloat16).
    skip_base_merge : bool, optional
        If True, skip merging with base model low-rank branches.
        Used for composed LoRAs that are already concatenated (default: False).

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Nunchaku format.
    """
    if isinstance(base_model, str):
        orig_state_dict = load_state_dict_in_safetensors(base_model)
    else:
        orig_state_dict = base_model

    if isinstance(lora, str):
        # Load the LoRA - check if it has transformer prefix
        temp_dict = load_state_dict_in_safetensors(lora)
        if any(k.startswith("transformer.") for k in temp_dict.keys()):
            # Standard Qwen Image LoRA with transformer prefix
            extra_lora_dict = filter_state_dict(temp_dict, filter_prefix="transformer.")
            # Remove the transformer. prefix after filtering
            renamed_dict = {}
            for k, v in extra_lora_dict.items():
                new_k = k.replace("transformer.", "") if k.startswith("transformer.") else k
                renamed_dict[new_k] = v
            extra_lora_dict = renamed_dict
        else:
            # LoRA without transformer prefix - use as is
            extra_lora_dict = temp_dict
    else:
        # When called from to_nunchaku, lora is already processed by to_diffusers
        extra_lora_dict = lora

    # Add transformer. prefix and rename blocks to match expectations
    renamed_dict = {}
    for k, v in extra_lora_dict.items():
        if not k.startswith("transformer."):
            renamed_dict[f"transformer.{k}"] = v
        else:
            renamed_dict[k] = v
    extra_lora_dict = renamed_dict

    # Convert each transformer block
    converted_dict = {}

    # Find all transformer block indices
    block_indices = set()
    for key in extra_lora_dict.keys():
        # Support both "transformer.blocks.X" and "transformer.transformer_blocks.X" formats
        if "transformer.blocks." in key or "transformer.transformer_blocks." in key:
            # Extract block index
            parts = key.split(".")
            for i, part in enumerate(parts):
                if (part == "blocks" or part == "transformer_blocks") and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        block_indices.add(block_idx)
                    except ValueError:
                        pass

    logger.debug(f"Converting {len(block_indices)} transformer blocks")

    for block_idx in sorted(block_indices):
        converted_block_name = f"transformer_blocks.{block_idx}"
        candidate_block_name = f"transformer.blocks.{block_idx}"

        block_dict = convert_to_nunchaku_transformer_block_lowrank_dict(
            orig_state_dict=orig_state_dict,
            extra_lora_dict=extra_lora_dict,
            converted_block_name=converted_block_name,
            candidate_block_name=candidate_block_name,
            default_dtype=default_dtype,
            skip_base_merge=skip_base_merge,
        )

        converted_dict.update(block_dict)

    return converted_dict


def to_nunchaku(
    input_lora: str | dict[str, torch.Tensor],
    quant_path: str | None = None,
    base_sd: dict[str, torch.Tensor] | None = None,
    output_path: str | None = None,
    default_dtype: torch.dtype = torch.bfloat16,
    skip_base_merge: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert Qwen Image LoRA weights to Nunchaku format.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a LoRA weight dictionary.
    quant_path : str, optional
        Path to quantized base model (for merging existing LoRA).
    base_sd : dict[str, torch.Tensor], optional
        Base model state dict (alternative to quant_path).
    output_path : str, optional
        If given, save the converted weights to this path.
    default_dtype : torch.dtype, optional
        Output tensor dtype (default: torch.bfloat16).
    skip_base_merge : bool, optional
        If True, skip merging with base model low-rank branches.
        Used for composed LoRAs that are already concatenated (default: False).

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Nunchaku format.

    Examples
    --------
    >>> lora_nunchaku = to_nunchaku("lora.safetensors", output_path="lora_nunchaku.safetensors")
    >>> lora_nunchaku = to_nunchaku(lora_dict, quant_path="quant_model.safetensors")
    >>> composed_lora = to_nunchaku(composed_dict, base_sd=base, skip_base_merge=True)
    """
    # Check if already in Nunchaku format
    if is_nunchaku_format(input_lora):
        logger.info("LoRA is already in Nunchaku format")
        if isinstance(input_lora, str):
            result = load_state_dict_in_safetensors(input_lora, device="cpu")
        else:
            result = input_lora
        if output_path is not None and isinstance(input_lora, str) and input_lora != output_path:
            save_file(result, output_path)
        return result

    # Check if already in Diffusers format (e.g., from compose_lora)
    if isinstance(input_lora, str):
        input_dict = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        input_dict = input_lora

    if is_diffusers_format(input_dict):
        logger.debug("LoRA is already in Diffusers format, skipping conversion")
        lora_diffusers = input_dict
    else:
        # Convert to Diffusers format first
        logger.debug("Converting to Diffusers format")
        lora_diffusers = to_diffusers(input_lora)

    # Load base model if provided
    if base_sd is not None:
        logger.debug("Using provided base model state dict")
        base_model = base_sd
    elif quant_path is not None:
        logger.debug(f"Loading base model from {quant_path}")
        base_model = load_state_dict_in_safetensors(quant_path)
    else:
        base_model = {}

    # Convert to Nunchaku format
    logger.debug("Converting to Nunchaku format")
    lora_nunchaku = convert_to_nunchaku_qwenimage_lowrank_dict(
        base_model=base_model,
        lora=lora_diffusers,
        default_dtype=default_dtype,
        skip_base_merge=skip_base_merge,
    )

    # Save if output path is provided
    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(lora_nunchaku, output_path)
        logger.info(f"Saved converted LoRA weights to {output_path}")

    return lora_nunchaku


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Qwen Image LoRA to Nunchaku format")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to the input LoRA safetensors file")
    parser.add_argument(
        "--quant-path", type=str, default=None, help="Path to the quantized base model safetensors file (optional)"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the converted LoRA in Nunchaku format"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for the converted weights",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    to_nunchaku(args.lora_path, quant_path=args.quant_path, output_path=args.output_path, default_dtype=dtype)

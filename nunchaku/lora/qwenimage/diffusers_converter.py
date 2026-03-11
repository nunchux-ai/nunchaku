"""
This module implements the functions to convert Qwen Image LoRA weights from various formats
to the Diffusers format, which will later be converted to Nunchaku format.
"""

import argparse
import logging
import os

import torch
from safetensors.torch import save_file

from ...utils import load_state_dict_in_safetensors

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def handle_kohya_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Kohya LoRA format keys to Diffusers format for Qwen Image.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights, possibly in Kohya format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    # Check if the state_dict is in the Kohya format
    # Kohya format typically has keys like: lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lora_down.weight
    if not any("lora_unet" in k or "lora_transformer" in k for k in state_dict.keys()):
        return state_dict

    converted_dict = {}
    for k, v in state_dict.items():
        new_k = k

        # Convert Kohya prefix to standard format
        if "lora_unet_" in k or "lora_transformer_" in k:
            new_k = new_k.replace("lora_unet_", "").replace("lora_transformer_", "")

            # Convert Kohya's lora_down/lora_up to lora_A/lora_B
            if ".lora_down." in new_k:
                new_k = new_k.replace(".lora_down.", ".lora_A.")
            elif ".lora_up." in new_k:
                new_k = new_k.replace(".lora_up.", ".lora_B.")

            # Convert underscores to dots for proper hierarchy
            # e.g., transformer_blocks_0_img_mlp -> transformer_blocks.0.img_mlp
            parts = new_k.split(".")
            if len(parts) > 0:
                # Process the first part (module path)
                module_path = parts[0]
                # Replace underscores with dots, but be careful with numbers
                import re

                # Match patterns like: transformer_blocks_0_img_mod
                module_path = re.sub(r"_(\d+)_", r".\1.", module_path)
                module_path = re.sub(r"_", ".", module_path)
                parts[0] = module_path
                new_k = ".".join(parts)

        converted_dict[new_k] = v

    logger.debug(f"Converted {len(state_dict)} Kohya LoRA keys")
    return converted_dict


def convert_peft_to_comfyui(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert PEFT LoRA format keys to ComfyUI format for Qwen Image.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in PEFT format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in ComfyUI format.
    """
    converted_dict = {}

    for k, v in state_dict.items():
        new_k = k

        # Remove PEFT prefix: base_model.model.
        if new_k.startswith("base_model.model."):
            new_k = new_k.replace("base_model.model.", "")

        # Convert PEFT adapter naming to standard LoRA naming
        # PEFT uses: module.lora_A.default.weight -> module.lora_A.weight
        new_k = new_k.replace(".lora_A.default.", ".lora_A.")
        new_k = new_k.replace(".lora_B.default.", ".lora_B.")
        new_k = new_k.replace(".lora_A.weight", ".lora_A.weight")
        new_k = new_k.replace(".lora_B.weight", ".lora_B.weight")

        converted_dict[new_k] = v

    logger.debug(f"Converted {len(state_dict)} PEFT LoRA keys")
    return converted_dict


def handle_qwen_to_diffusers_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Ensure Qwen Image LoRA is in proper Diffusers format.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights for Qwen Image.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    converted_dict = {}
    duplicate_keys = []

    for k, v in state_dict.items():
        new_k = k

        # Remove .default. from PEFT-style naming (e.g., lora_A.default.weight -> lora_A.weight)
        # This handles LoRAs trained with PEFT/Kohya without base_model.model. prefix
        new_k = new_k.replace(".lora_A.default.", ".lora_A.")
        new_k = new_k.replace(".lora_B.default.", ".lora_B.")
        # Also handle if it's at the end before weight
        if ".default.weight" in new_k:
            new_k = new_k.replace(".default.weight", ".weight")

        # Ensure transformer_blocks prefix exists
        if not new_k.startswith("transformer_blocks.") and not new_k.startswith("transformer."):
            # If it starts with blocks., add transformer. prefix
            if new_k.startswith("blocks."):
                new_k = "transformer." + new_k
            # If it doesn't have transformer prefix at all, add it
            elif "transformer_blocks." in new_k:
                pass  # Already has the right format
            else:
                # Add transformer. prefix
                new_k = "transformer." + new_k

        # Normalize transformer_blocks vs transformer.blocks
        if new_k.startswith("transformer_blocks."):
            new_k = "transformer." + new_k.replace("transformer_blocks.", "blocks.")

        # Unify various LoRA naming conventions to lora_A/lora_B
        # Support multiple formats:
        # - Nunchaku: lora_down / lora_up (no dots, no .weight)
        # - Standard: .lora_down. / .lora_up.
        # - Alternative: .lora.down. / .lora.up.
        # This ensures consistent naming for downstream composition

        # Check for Nunchaku format (ends with lora_down or lora_up, no .weight)
        if new_k.endswith(".lora_down") or ".lora_down." in new_k:
            if new_k.endswith(".lora_down"):
                new_k = new_k.replace(".lora_down", ".lora_A.weight")
            else:
                new_k = new_k.replace(".lora_down.", ".lora_A.")
        elif new_k.endswith(".lora_up") or ".lora_up." in new_k:
            if new_k.endswith(".lora_up"):
                new_k = new_k.replace(".lora_up", ".lora_B.weight")
            else:
                new_k = new_k.replace(".lora_up.", ".lora_B.")
        elif ".lora.down." in new_k:
            new_k = new_k.replace(".lora.down.", ".lora_A.")
        elif ".lora.up." in new_k:
            new_k = new_k.replace(".lora.up.", ".lora_B.")

        # Check for duplicate keys before adding
        if new_k in converted_dict:
            duplicate_keys.append((k, new_k, v.shape, converted_dict[new_k].shape))
            # Keep the one with larger rank (for backward compatibility)
            existing_rank = converted_dict[new_k].shape[0] if converted_dict[new_k].ndim >= 2 else 0
            new_rank = v.shape[0] if v.ndim >= 2 else 0

            if new_rank > existing_rank:
                # New one has larger rank, replace it
                converted_dict[new_k] = v
            # else: keep existing (larger rank)
        else:
            converted_dict[new_k] = v

    # Report duplicate keys if any
    if duplicate_keys:
        logger.debug(f"Found {len(duplicate_keys)} duplicate keys in handle_qwen_to_diffusers_format:")
        for orig_k, new_k, new_shape, old_shape in duplicate_keys[:5]:
            logger.debug(f"   {orig_k} -> {new_k}: new_shape={new_shape}, old_shape={old_shape}")

    return converted_dict


def unpack_nunchaku_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Unpack Nunchaku packed LoRA format to standard Diffusers format.

    Nunchaku format uses packed lora_down/lora_up tensors.
    This function unpacks them and converts to lora_A.weight/lora_B.weight format.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in Nunchaku packed format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    from .packer import unpack_lowrank_weight

    unpacked_dict = {}

    # Check if this is truly packed format or just using lora_down/lora_up naming
    # Packed format has swapped dimensions, unpacked format has normal dimensions
    sample_down_key = next(
        (
            k
            for k in state_dict.keys()
            if ".to_qkv.lora_down" in k or ".add_qkv_proj.lora_down" in k or "attn" in k and ".lora_down" in k
        ),
        None,
    )
    needs_unpack = True

    if sample_down_key:
        sample_shape = state_dict[sample_down_key].shape

        # For lora_down, expected unpacked shape is [rank, in_features]
        # For Qwen Image, in_features=3072, rank is usually 64-128
        # If shape[0] < shape[1], it's likely already unpacked
        if sample_shape[0] < sample_shape[1]:
            needs_unpack = False
            logger.debug(f"LoRA format: already unpacked [rank={sample_shape[0]}, in_features={sample_shape[1]}]")

    for k, v in state_dict.items():
        if k.endswith(".lora_down") or ".lora_down." in k:
            # Convert dtype if needed
            if v.dtype not in (torch.float16, torch.bfloat16):
                v = v.to(torch.bfloat16)

            # Unpack only if needed
            if needs_unpack:
                unpacked_v = unpack_lowrank_weight(v, down=True)
            else:
                unpacked_v = v  # Already unpacked, keep as-is

            # Convert to Diffusers naming
            new_k = (
                k.replace(".lora_down", ".lora_A.weight")
                if k.endswith(".lora_down")
                else k.replace(".lora_down.", ".lora_A.")
            )
            unpacked_dict[new_k] = unpacked_v
        elif k.endswith(".lora_up") or ".lora_up." in k:
            # Convert dtype if needed
            if v.dtype not in (torch.float16, torch.bfloat16):
                v = v.to(torch.bfloat16)

            # Unpack only if needed
            if needs_unpack:
                unpacked_v = unpack_lowrank_weight(v, down=False)
            else:
                unpacked_v = v  # Already unpacked, keep as-is

            # Convert to Diffusers naming
            new_k = (
                k.replace(".lora_up", ".lora_B.weight")
                if k.endswith(".lora_up")
                else k.replace(".lora_up.", ".lora_B.")
            )
            unpacked_dict[new_k] = unpacked_v
        else:
            # Keep other tensors as-is (also convert dtype if needed)
            if v.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                v = v.to(torch.bfloat16)
            unpacked_dict[k] = v

    logger.debug(f"Unpacked {len(state_dict)} Nunchaku LoRA keys")
    return unpacked_dict


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    """
    Convert Qwen Image LoRA weights to Diffusers format, which will later be converted to Nunchaku format.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a LoRA weight dictionary.
    output_path : str, optional
        If given, save the converted weights to this path.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    # Check if this is Nunchaku packed format and unpack if needed
    from .utils import is_nunchaku_format

    if is_nunchaku_format(tensors):
        logger.debug("Detected Nunchaku packed format, unpacking...")
        tensors = unpack_nunchaku_lora(tensors)

    # First, try to detect and convert Kohya format
    tensors = handle_kohya_lora(tensors)

    # Convert FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

    # Apply PEFT-specific key conversion
    if any(k.startswith("base_model.model.") for k in tensors.keys()):
        logger.debug("Converting PEFT format to ComfyUI format")
        tensors = convert_peft_to_comfyui(tensors)

    # Ensure proper Diffusers format for Qwen Image
    tensors = handle_qwen_to_diffusers_format(tensors)

    # Handle .alpha parameters for LoRA scaling
    # Extract and apply alpha scaling to lora_A weights
    alpha_dict = {}
    for k in list(tensors.keys()):
        if ".alpha" in k:
            v = tensors.pop(k)
            # Convert to scalar if it's a 0D tensor
            alpha_value = v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v
            alpha_dict[k] = alpha_value
    
    # Apply alpha scaling to lora_A weights
    for k in list(tensors.keys()):
        if "lora_A" in k and tensors[k].ndim == 2:
            # Find corresponding alpha key
            base_key = k.replace(".lora_A.weight", "").replace(".weight", "")
            alpha_key = f"{base_key}.alpha"
            if alpha_key in alpha_dict:
                alpha = alpha_dict[alpha_key]
                rank = tensors[k].shape[0]
                alpha_scale = alpha / rank
                tensors[k] = tensors[k] * alpha_scale
                logger.debug(f"Applied alpha scaling to {k}: alpha={alpha}, rank={rank}, scale={alpha_scale}")

    # Conversion complete

    # Save if output path is provided
    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(tensors, output_path)
        logger.info(f"Saved converted LoRA weights to {output_path}")

    return tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen Image LoRA to Diffusers format")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to the input LoRA safetensors file")
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the converted LoRA in Diffusers format"
    )
    args = parser.parse_args()

    to_diffusers(args.lora_path, args.output_path)

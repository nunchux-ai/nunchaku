"""
Compose multiple LoRA weights into a single LoRA for Qwen Image models.

This script merges several LoRA safetensors files into one, applying individual strength values to each.

**Example Usage:**

.. code-block:: bash

    python -m nunchaku.lora.qwenimage.compose \\
        -i lora1.safetensors lora2.safetensors \\
        -s 0.8 1.0 \\
        -o composed_lora.safetensors

**Arguments:**

- ``-i``, ``--input-paths``: Input LoRA safetensors files (one or more).
- ``-s``, ``--strengths``: Strength value for each LoRA (must match number of inputs).
- ``-o``, ``--output-path``: Output path for the composed LoRA safetensors file.

This will merge ``lora1.safetensors`` (strength 0.8) and ``lora2.safetensors`` (strength 1.0) into ``composed_lora.safetensors``.

**Main Function**

:func:`compose_lora`
"""

import argparse
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .diffusers_converter import to_diffusers
from .utils import is_nunchaku_format


def normalize_lora_keys(lora: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize LoRA keys to standard format: transformer.blocks.X...

    This ensures all LoRAs use the same naming convention for proper composition,
    regardless of their original source format.

    Handles various formats:
    - diffusion_model.transformer_blocks.X → transformer.blocks.X
    - transformer_blocks.X → transformer.blocks.X
    - blocks.X → transformer.blocks.X
    - transformer.blocks.X → transformer.blocks.X (no change)

    Parameters
    ----------
    lora : dict[str, torch.Tensor]
        LoRA weights dictionary with potentially mixed key formats.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights with standardized key names.

    Examples
    --------
    >>> lora = {"diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": tensor}
    >>> normalized = normalize_lora_keys(lora)
    >>> # Result: {"transformer.blocks.0.attn.to_q.lora_A.weight": tensor}
    """
    normalized = {}
    for k, v in lora.items():
        new_k = k

        # Remove diffusion_model prefix (common in ComfyUI LoRAs)
        if new_k.startswith("diffusion_model."):
            new_k = new_k.replace("diffusion_model.", "")

        # Normalize transformer_blocks → transformer.blocks
        if new_k.startswith("transformer_blocks."):
            new_k = new_k.replace("transformer_blocks.", "transformer.blocks.")
        elif new_k.startswith("blocks.") and not new_k.startswith("transformer."):
            # Add transformer. prefix if missing
            new_k = "transformer." + new_k
        elif not new_k.startswith("transformer."):
            # If no transformer prefix at all, add it
            if "transformer_blocks." in new_k:
                # Handle cases like: xxx.transformer_blocks.0...
                new_k = new_k.replace("transformer_blocks.", "transformer.blocks.")
            elif ".blocks." in new_k and "transformer" not in new_k:
                # Handle cases like: xxx.blocks.0...
                parts = new_k.split(".blocks.")
                new_k = parts[0] + ".transformer.blocks." + parts[1] if len(parts) > 1 else new_k

        normalized[new_k] = v

    return normalized


def compose_lora(
    loras: list[tuple[str | dict[str, torch.Tensor], float]], output_path: str | None = None
) -> dict[str, torch.Tensor]:
    """
    Compose multiple Qwen Image LoRA weights into a single LoRA representation.

    Parameters
    ----------
    loras : list of (str or dict[str, torch.Tensor], float)
        Each tuple contains:
            - Path to a LoRA safetensors file or a LoRA weights dictionary.
            - Strength/scale factor for that LoRA.
    output_path : str, optional
        Path to save the composed LoRA weights as a safetensors file. If None, does not save.

    Returns
    -------
    dict[str, torch.Tensor]
        The composed LoRA weights.

    Raises
    ------
    AssertionError
        If LoRA weights are in Nunchaku format (must be converted to Diffusers format first)
        or if tensor shapes are incompatible.

    Notes
    -----
    - Converts all input LoRAs to Diffusers format.
    - Handles QKV projection fusion for attention layers.
    - Applies strength scaling to LoRA weights.
    - Concatenates multiple LoRAs along appropriate dimensions.
    - Handles dual-stream architecture (img and txt streams).

    Examples
    --------
    >>> lora_paths = [("lora1.safetensors", 0.8), ("lora2.safetensors", 0.6)]
    >>> composed = compose_lora(lora_paths, "composed_lora.safetensors")
    >>> lora_dicts = [({"layer.weight": torch.randn(10, 20)}, 1.0)]
    >>> composed = compose_lora(lora_dicts)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Disable early return to force QKV fusion
    # if len(loras) == 1:
    #     logger.info(f"Single LoRA detected, strength={loras[0][1]}")
    #     is_nunchaku = is_nunchaku_format(loras[0][0])
    #     logger.info(f"is_nunchaku_format={is_nunchaku}")
    #     if is_nunchaku and (loras[0][1] - 1) < 1e-5:
    #         logger.info("Early return: LoRA is already in Nunchaku format with strength 1.0")
    #         if isinstance(loras[0][0], str):
    #             return load_state_dict_in_safetensors(loras[0][0], device="cpu")
    #         else:
    #             return loras[0][0]
    #     logger.info("Proceeding to compose logic...")

    import logging

    logger = logging.getLogger(__name__)

    logger.debug(f"Composing {len(loras)} LoRAs...")

    AMPLIFICATION_FACTOR = 1.0

    composed = {}
    for idx, (lora, strength) in enumerate(loras):
        # Apply amplification factor to compensate for quantization loss
        amplified_strength = strength * AMPLIFICATION_FACTOR
        logger.debug(
            f"LoRA {idx+1}: strength {strength} → {amplified_strength} (amplification: {AMPLIFICATION_FACTOR}x)"
        )

        # Auto-convert Nunchaku format to Diffusers format if needed
        if is_nunchaku_format(lora):
            lora = to_diffusers(lora)
        else:
            lora = to_diffusers(lora)

        # Normalize all keys to standard format (transformer.blocks.X...)
        # This ensures different LoRAs can be properly concatenated
        lora = normalize_lora_keys(lora)

        # Extract and remove .alpha parameters (LoRA scaling factors)
        alpha_dict = {}
        for k in list(lora.keys()):
            if ".alpha" in k:
                v = lora.pop(k)
                # Convert to scalar if it's a 0D tensor
                alpha_value = v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v
                alpha_dict[k] = alpha_value

        if len(alpha_dict) > 0:
            logger.debug(f"Found {len(alpha_dict)} alpha scaling factors")

        # Apply alpha scaling directly to lora dictionary
        # This ensures all lora_A weights are consistently scaled before processing
        for k in list(lora.keys()):
            if "lora_A" in k and lora[k].ndim == 2:
                # Find corresponding alpha key
                base_key = k.replace(".lora_A.weight", "").replace(".weight", "")
                alpha_key = f"{base_key}.alpha"
                if alpha_key in alpha_dict:
                    alpha = alpha_dict[alpha_key]
                    rank = lora[k].shape[0]
                    alpha_scale = alpha / rank
                    lora[k] = lora[k] * alpha_scale  # Modify in-place in the dictionary
                    logger.debug(f"  Applied alpha scaling to {k}: alpha={alpha}, rank={rank}, scale={alpha_scale}")

        for k, v in list(lora.items()):

            # Handle 1D tensors (bias, normalization parameters)
            if v.ndim == 1:
                previous_tensor = composed.get(k, None)
                if previous_tensor is None:
                    # For normalization layers, don't apply strength
                    if "norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k:
                        composed[k] = v
                    else:
                        composed[k] = v * amplified_strength
                else:
                    # Normalization layers should not be accumulated
                    assert not ("norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k)
                    composed[k] = previous_tensor + v * amplified_strength
            else:
                assert v.ndim == 2, f"Expected 2D tensor, got {v.ndim}D for key {k}"

                # Handle QKV fusion for attention layers
                # Qwen Image has both to_q/to_k/to_v and add_q_proj/add_k_proj/add_v_proj
                # Also handle already-fused to_qkv/add_qkv_proj from Nunchaku format
                if (".to_qkv." in k or ".add_qkv_proj." in k) and ("lora_A" in k or "lora_B" in k):
                    # Already fused - apply strength to lora_A and concatenate both
                    if "lora_A" in k:
                        v = v * amplified_strength

                    previous_lora = composed.get(k, None)

                    if previous_lora is None:
                        composed[k] = v
                    else:
                        # Concatenate along appropriate dimension
                        if "lora_A" in k:
                            composed[k] = torch.cat([previous_lora, v], dim=0)
                        else:  # lora_B
                            composed[k] = torch.cat([previous_lora, v], dim=1)

                elif ".to_q." in k or ".add_q_proj." in k:
                    if "lora_B" in k:
                        continue

                    # Get Q, K, V weights
                    q_a = v
                    k_a = lora[k.replace(".to_q.", ".to_k.").replace(".add_q_proj.", ".add_k_proj.")]
                    v_a = lora[k.replace(".to_q.", ".to_v.").replace(".add_q_proj.", ".add_v_proj.")]

                    q_b = lora[k.replace("lora_A", "lora_B")]
                    k_b = lora[
                        k.replace("lora_A", "lora_B")
                        .replace(".to_q.", ".to_k.")
                        .replace(".add_q_proj.", ".add_k_proj.")
                    ]
                    v_b = lora[
                        k.replace("lora_A", "lora_B")
                        .replace(".to_q.", ".to_v.")
                        .replace(".add_q_proj.", ".add_v_proj.")
                    ]

                    # Add padding if ranks are different
                    max_rank = max(q_a.shape[0], k_a.shape[0], v_a.shape[0])
                    q_a = F.pad(q_a, (0, 0, 0, max_rank - q_a.shape[0]))
                    k_a = F.pad(k_a, (0, 0, 0, max_rank - k_a.shape[0]))
                    v_a = F.pad(v_a, (0, 0, 0, max_rank - v_a.shape[0]))
                    q_b = F.pad(q_b, (0, max_rank - q_b.shape[1]))
                    k_b = F.pad(k_b, (0, max_rank - k_b.shape[1]))
                    v_b = F.pad(v_b, (0, max_rank - v_b.shape[1]))

                    # Check if Q, K, V share the same lora_A (common case)
                    # For QKV fusion:
                    # - lora_A (down projection): Q, K, V should share the same input space, so use q_a
                    # - lora_B (up projection): Concatenate along output dimension (dim=0)
                    if torch.isclose(q_a, k_a).all() and torch.isclose(q_a, v_a).all():
                        lora_a = q_a
                        lora_b = torch.cat((q_b, k_b, v_b), dim=0)
                    else:
                        # Different lora_A for Q, K, V - average them
                        # lora_A shape: [rank, in_features]
                        # Since Q, K, V share the same input space, we average their lora_A
                        lora_a = (q_a + k_a + v_a) / 3.0

                        # lora_B shape: [out_features, rank]
                        # For QKV fusion, concatenate along out_features dimension (dim=0)
                        # Result: [out_features * 3, rank]
                        lora_b = torch.cat((q_b, k_b, v_b), dim=0)

                    # Apply amplified strength to lora_A
                    lora_a = lora_a * amplified_strength

                    # Verify lora_a and lora_b ranks match
                    if lora_a.shape[0] != lora_b.shape[1]:
                        logger.error(
                            f"❌ QKV fusion rank mismatch: lora_a rank={lora_a.shape[0]}, lora_b rank={lora_b.shape[1]}"
                        )
                        logger.error(f"   Key: {k}")
                        logger.error(f"   q_a={q_a.shape}, k_a={k_a.shape}, v_a={v_a.shape}")
                        logger.error(f"   q_b={q_b.shape}, k_b={k_b.shape}, v_b={v_b.shape}")
                        logger.error(f"   lora_a={lora_a.shape}, lora_b={lora_b.shape}")

                    # Create fused key names
                    new_k_a = k.replace(".to_q.", ".to_qkv.").replace(".add_q_proj.", ".add_qkv_proj.")
                    new_k_b = new_k_a.replace("lora_A", "lora_B")

                    # Concatenate with previous LoRAs
                    for kk, vv, dim in ((new_k_a, lora_a, 0), (new_k_b, lora_b, 1)):
                        previous_lora = composed.get(kk, None)

                        if previous_lora is not None:
                            # Verify non-concatenation dimension matches
                            non_cat_dim = 1 if dim == 0 else 0
                            if previous_lora.shape[non_cat_dim] != vv.shape[non_cat_dim]:
                                raise ValueError(
                                    f"Cannot compose LoRAs with incompatible shapes for key '{kk}':\n"
                                    f"  Previous LoRA: shape={previous_lora.shape}\n"
                                    f"  Current LoRA: shape={vv.shape}\n"
                                    f"  Dimension {non_cat_dim} must match but got {previous_lora.shape[non_cat_dim]} vs {vv.shape[non_cat_dim]}\n"
                                    f"  These LoRAs have incompatible architectures and cannot be concatenated."
                                )
                            composed[kk] = torch.cat([previous_lora, vv], dim=dim)
                        else:
                            composed[kk] = vv

                elif ".to_k." in k or ".to_v." in k or ".add_k_proj." in k or ".add_v_proj." in k:
                    # Skip K and V as they're already handled with Q
                    continue
                else:
                    # Handle other layers (img_mlp, txt_mlp, img_mod, txt_mod, etc.)
                    if "lora_A" in k:
                        v = v * amplified_strength

                    previous_lora = composed.get(k, None)

                    if previous_lora is None:
                        composed[k] = v
                    else:
                        # Concatenate along appropriate dimension
                        if "lora_A" in k:
                            # Check for dimension mismatch (rare but possible)
                            if previous_lora.shape[1] != v.shape[1]:
                                # Handle dimension expansion (similar to Flux x_embedder handling)
                                expanded_size = max(previous_lora.shape[1], v.shape[1])
                                if expanded_size > previous_lora.shape[1]:
                                    expanded_previous_lora = torch.zeros(
                                        (previous_lora.shape[0], expanded_size),
                                        device=previous_lora.device,
                                        dtype=previous_lora.dtype,
                                    )
                                    expanded_previous_lora[:, : previous_lora.shape[1]] = previous_lora
                                else:
                                    expanded_previous_lora = previous_lora
                                if expanded_size > v.shape[1]:
                                    expanded_v = torch.zeros(
                                        (v.shape[0], expanded_size), device=v.device, dtype=v.dtype
                                    )
                                    expanded_v[:, : v.shape[1]] = v
                                else:
                                    expanded_v = v
                                composed[k] = torch.cat([expanded_previous_lora, expanded_v], dim=0)
                            else:
                                composed[k] = torch.cat([previous_lora, v], dim=0)
                        else:  # lora_B
                            # Verify dimension compatibility before concatenation
                            if previous_lora.shape[0] != v.shape[0]:
                                raise ValueError(
                                    f"Cannot compose LoRAs with incompatible shapes for key '{k}':\n"
                                    f"  Previous LoRA: shape={previous_lora.shape}\n"
                                    f"  Current LoRA: shape={v.shape}\n"
                                    f"  Dimension 0 (output features) must match but got {previous_lora.shape[0]} vs {v.shape[0]}\n"
                                    f"  These LoRAs have incompatible output dimensions and cannot be concatenated."
                                )
                            composed[k] = torch.cat([previous_lora, v], dim=1)

    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(composed, output_path)
    # Summary log
    sample_key = next((k for k in composed.keys() if "to_qkv.lora_A" in k), None)

    if sample_key:
        rank = composed[sample_key].shape[0]
        logger.info(f"Composed {len(loras)} LoRAs: {len(composed)} keys, rank={rank}")
    else:
        logger.info(f"Composed {len(loras)} LoRAs: {len(composed)} keys")

    # Remove diffusion_model prefix for compatibility with to_nunchaku
    normalized = {}
    for k, v in composed.items():
        # Remove diffusion_model. prefix if present
        if k.startswith("diffusion_model."):
            k = k.replace("diffusion_model.", "", 1)
        normalized[k] = v

    return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose multiple Qwen Image LoRAs")
    parser.add_argument(
        "-i", "--input-paths", type=str, nargs="*", required=True, help="Paths to the LoRA safetensors files"
    )
    parser.add_argument("-s", "--strengths", type=float, nargs="*", required=True, help="Strengths for each LoRA")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to the output safetensors file")
    args = parser.parse_args()
    assert len(args.input_paths) == len(args.strengths), "Number of input paths must match number of strengths"
    compose_lora(list(zip(args.input_paths, args.strengths)), args.output_path)

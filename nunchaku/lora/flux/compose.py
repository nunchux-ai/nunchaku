"""
Compose multiple LoRA weights into a single LoRA for FLUX models.

This script merges several LoRA safetensors files into one, applying individual strength values to each.

**Example Usage:**

.. code-block:: bash

    python -m nunchaku.lora.flux.compose \\
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
from .utils import is_nunchaku_format, load_state_dict_in_safetensors


def compose_lora(
    loras: list[tuple[str | dict[str, torch.Tensor], float]], output_path: str | None = None
) -> dict[str, torch.Tensor]:
    """
    Compose multiple LoRA weights into a single LoRA representation.

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
    - Handles normalization layers, bias vectors, and FLUX.1-tools LoRA compatibility.

    Examples
    --------
    >>> lora_paths = [("lora1.safetensors", 0.8), ("lora2.safetensors", 0.6)]
    >>> composed = compose_lora(lora_paths, "composed_lora.safetensors")
    >>> lora_dicts = [({"layer.weight": torch.randn(10, 20)}, 1.0)]
    >>> composed = compose_lora(lora_dicts)
    """
    if len(loras) == 1:
        if is_nunchaku_format(loras[0][0]) and (loras[0][1] - 1) < 1e-5:
            if isinstance(loras[0][0], str):
                return load_state_dict_in_safetensors(loras[0][0], device="cpu")
            else:
                return loras[0][0]

    composed = {}
    for lora, strength in loras:
        assert not is_nunchaku_format(lora)
        lora = to_diffusers(lora)
        for k, v in list(lora.items()):
            if v.ndim == 1:
                previous_tensor = composed.get(k, None)
                if previous_tensor is None:
                    if "norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k:
                        composed[k] = v
                    else:
                        composed[k] = v * strength
                else:
                    assert not ("norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k)
                    composed[k] = previous_tensor + v * strength
            else:
                assert v.ndim == 2
                if ".to_q." in k or ".add_q_proj." in k:  # qkv must all exist
                    if "lora_B" in k:
                        continue

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

                    # Add paddings if their ranks are different
                    max_rank = max(q_a.shape[0], k_a.shape[0], v_a.shape[0])
                    q_a = F.pad(q_a, (0, 0, 0, max_rank - q_a.shape[0]))
                    k_a = F.pad(k_a, (0, 0, 0, max_rank - k_a.shape[0]))
                    v_a = F.pad(v_a, (0, 0, 0, max_rank - v_a.shape[0]))
                    q_b = F.pad(q_b, (0, max_rank - q_b.shape[1]))
                    k_b = F.pad(k_b, (0, max_rank - k_b.shape[1]))
                    v_b = F.pad(v_b, (0, max_rank - v_b.shape[1]))

                    if torch.isclose(q_a, k_a).all() and torch.isclose(q_a, v_a).all():
                        lora_a = q_a
                        lora_b = torch.cat((q_b, k_b, v_b), dim=0)
                    else:
                        lora_a_group = (q_a, k_a, v_a)
                        new_shape_a = [sum([_.shape[0] for _ in lora_a_group]), q_a.shape[1]]
                        lora_a = torch.zeros(new_shape_a, dtype=q_a.dtype, device=q_a.device)
                        start_dim = 0
                        for tensor in lora_a_group:
                            lora_a[start_dim : start_dim + tensor.shape[0]] = tensor
                            start_dim += tensor.shape[0]

                        lora_b_group = (q_b, k_b, v_b)
                        new_shape_b = [sum([_.shape[0] for _ in lora_b_group]), sum([_.shape[1] for _ in lora_b_group])]
                        lora_b = torch.zeros(new_shape_b, dtype=q_b.dtype, device=q_b.device)
                        start_dims = (0, 0)
                        for tensor in lora_b_group:
                            end_dims = (start_dims[0] + tensor.shape[0], start_dims[1] + tensor.shape[1])
                            lora_b[start_dims[0] : end_dims[0], start_dims[1] : end_dims[1]] = tensor
                            start_dims = end_dims

                    lora_a = lora_a * strength

                    new_k_a = k.replace(".to_q.", ".to_qkv.").replace(".add_q_proj.", ".add_qkv_proj.")
                    new_k_b = new_k_a.replace("lora_A", "lora_B")

                    for kk, vv, dim in ((new_k_a, lora_a, 0), (new_k_b, lora_b, 1)):
                        previous_lora = composed.get(kk, None)
                        composed[kk] = vv if previous_lora is None else torch.cat([previous_lora, vv], dim=dim)

                elif ".to_k." in k or ".to_v." in k or ".add_k_proj." in k or ".add_v_proj." in k:
                    continue
                else:
                    if "lora_A" in k:
                        v = v * strength

                    previous_lora = composed.get(k, None)
                    if previous_lora is None:
                        composed[k] = v
                    else:
                        if "lora_A" in k:
                            if previous_lora.shape[1] != v.shape[1]:  # flux.1-tools LoRA compatibility
                                assert "x_embedder" in k
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
                        else:
                            composed[k] = torch.cat([previous_lora, v], dim=1)
                    """
                    composed[k] = (
                        v if previous_lora is None else torch.cat([previous_lora, v], dim=0 if "lora_A" in k else 1)
                    )
                    """
    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(composed, output_path)
    return composed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-paths", type=str, nargs="*", required=True, help="paths to the lora safetensors files"
    )
    parser.add_argument("-s", "--strengths", type=float, nargs="*", required=True, help="strengths for each lora")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="path to the output safetensors file")
    args = parser.parse_args()
    assert len(args.input_paths) == len(args.strengths)
    compose_lora(list(zip(args.input_paths, args.strengths)), args.output_path)

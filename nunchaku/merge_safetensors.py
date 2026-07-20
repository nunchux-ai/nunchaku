"""
Merge split safetensors model files (deprecated format) into a single safetensors file.

**Example usage**

.. code-block:: bash

    python -m nunchaku.merge_safetensors -i <input_path_or_repo> -o <output_path>

**Arguments**

- ``-i``, ``--input-path`` (Path): Path to the model directory or HuggingFace repo.
- ``-o``, ``--output-path`` (Path): Path to save the merged safetensors file.

It will combine the ``unquantized_layers.safetensors`` and ``transformer_blocks.safetensors``
files (and associated config files) from a local directory or a HuggingFace Hub repository
into a single safetensors file with appropriate metadata.

**Main Function**

:func:`merge_safetensors`
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import constants, hf_hub_download
from safetensors.torch import save_file

from .utils import load_state_dict_in_safetensors


def merge_safetensors(
    pretrained_model_name_or_path: str | os.PathLike[str], model_class: str, **kwargs
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    Merge split safetensors model files into a single state dict and metadata.

    This function loads the ``unquantized_layers.safetensors`` and ``transformer_blocks.safetensors``
    files (and associated config files) from a local directory or a HuggingFace Hub repository,
    and merges them into a single state dict and metadata dictionary.

    Parameters
    ----------
    pretrained_model_name_or_path : str or os.PathLike
        Path to the model directory or HuggingFace repo.
    model_class : str
        Specify model class. E.g. NunchakuFluxTransformer2dModel, NunchakuZImageTransformer2DModel or NunchakuKrea2Transformer2DModel
    **kwargs
        Additional keyword arguments for subfolder, comfy_config_path, and HuggingFace download options.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, str]]
        The merged state dict and metadata dictionary.

        - **state_dict**: The merged model state dict.
        - **metadata**: Dictionary containing ``config``, ``comfy_config``, ``model_class``, and ``quantization_config``.
    """
    subfolder = kwargs.get("subfolder", None)
    comfy_config_path = kwargs.get("comfy_config_path", None)

    if isinstance(pretrained_model_name_or_path, str):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
    if pretrained_model_name_or_path.exists():
        dirpath = pretrained_model_name_or_path if subfolder is None else pretrained_model_name_or_path / subfolder
        unquantized_part_path = dirpath / "unquantized_layers.safetensors"
        transformer_block_path = dirpath / "transformer_blocks.safetensors"
        config_path = dirpath / "config.json"
        if comfy_config_path is None:
            comfy_config_path = dirpath / "comfy_config.json"
    else:
        download_kwargs = {
            "subfolder": subfolder,
            "repo_type": "model",
            "revision": kwargs.get("revision", None),
            "cache_dir": kwargs.get("cache_dir", None),
            "local_dir": kwargs.get("local_dir", None),
            "user_agent": kwargs.get("user_agent", None),
            "force_download": kwargs.get("force_download", False),
            "proxies": kwargs.get("proxies", None),
            "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
            "token": kwargs.get("token", None),
            "local_files_only": kwargs.get("local_files_only", None),
            "headers": kwargs.get("headers", None),
            "endpoint": kwargs.get("endpoint", None),
            "resume_download": kwargs.get("resume_download", None),
            "force_filename": kwargs.get("force_filename", None),
            "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
        }
        unquantized_part_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="unquantized_layers.safetensors", **download_kwargs
        )
        transformer_block_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="transformer_blocks.safetensors", **download_kwargs
        )
        config_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="config.json", **download_kwargs
        )
        comfy_config_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="comfy_config.json", **download_kwargs
        )

    unquantized_part_sd = load_state_dict_in_safetensors(unquantized_part_path)
    transformer_block_sd = load_state_dict_in_safetensors(transformer_block_path)
    state_dict = unquantized_part_sd
    state_dict.update(transformer_block_sd)

    rank = next((v.shape[1] for k, v in transformer_block_sd.items() if ".lora_down" in k), 32)
    if "ZImage" in model_class:
        rank = next((v.shape[1] for k, v in transformer_block_sd.items() if ".proj_down" in k), 32)
        skip_refiners = not any(("refiner" in k and "attention.to_qkv" in k) for k in transformer_block_sd.keys())

    precision = "int4"
    for v in state_dict.values():
        assert isinstance(v, torch.Tensor)
        if v.dtype in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e8m0fnu,
        ]:
            precision = "fp4"
    quantization_config = {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": "fp8_e4m3_nan" if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
        "rank": rank,
    }
    if "ZImage" in model_class:
        quantization_config["skip_refiners"] = skip_refiners
    return state_dict, {
        "config": Path(config_path).read_text(),
        "comfy_config": Path(comfy_config_path).read_text(),
        "model_class": model_class,
        "quantization_config": json.dumps(quantization_config),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to model directory. It can also be a huggingface repo.",
    )
    parser.add_argument(
        "-m",
        "--model-class",
        type=str,
        required=True,
        help="Specify model class. E.g. NunchakuFluxTransformer2dModel, NunchakuZImageTransformer2DModel or NunchakuKrea2Transformer2DModel",
    )
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="Path to output path")
    args = parser.parse_args()
    state_dict, metadata = merge_safetensors(args.input_path, args.model_class)
    output_path = Path(args.output_path)
    print(f"    --input-path: {args.input_path}")
    print(f"    --model-class: {args.model_class}")
    print(f"    --output-path: {args.output_path}")
    dirpath = output_path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata=metadata)

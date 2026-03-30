from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from ...torch_transfer_utils import normalize_device, resolve_pin_memory
from .linear import W4Linear

__all__ = [
    "BaseNunchakuQwenEncoderModel",
    "collect_qwen_quantized_prefixes",
    "collect_required_qwen_quantized_prefixes",
    "NunchakuQwenCheckpointMetadata",
    "build_full_qwen_config_dict",
    "load_qwen_checkpoint_metadata",
    "load_qwen_runtime_state_dict",
    "materialize_meta_module_tensors",
    "patch_qwen_runtime_linears",
    "register_rotary_materializer",
    "restore_qwen_runtime_model_tensors",
    "resolve_checkpoint_path",
]

# Keep the old name importable for backward compatibility.
BaseNunchakuQwenTextEncoderModel: tp.TypeAlias = "BaseNunchakuQwenEncoderModel"


# ---------------------------------------------------------------------------
# P1: Rotary buffer materializer registry
# ---------------------------------------------------------------------------

RotaryMaterializerFn = tp.Callable[[nn.Module, str, torch.device], torch.Tensor | None]
_ROTARY_MATERIALIZERS: dict[str, RotaryMaterializerFn] = {}


def register_rotary_materializer(class_name: str):
    """Decorator that registers a rotary-buffer materializer for *class_name*."""
    def decorator(fn: RotaryMaterializerFn) -> RotaryMaterializerFn:
        _ROTARY_MATERIALIZERS[class_name] = fn
        return fn
    return decorator


@register_rotary_materializer("Qwen2_5_VisionRotaryEmbedding")
def _materialize_vision_rotary(owner: nn.Module, name: str, device: torch.device) -> torch.Tensor | None:
    if name != "inv_freq":
        return None
    dim = int(getattr(owner, "dim"))
    theta = float(getattr(owner, "theta"))
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))


def _materialize_rope_with_init_fn(owner: nn.Module, name: str, device: torch.device) -> torch.Tensor | None:
    """Shared materializer for RoPE embeddings that use ``compute_default_rope_parameters``."""
    if name not in ("inv_freq", "original_inv_freq"):
        return None
    config = getattr(owner, "config")
    rope_type = str(getattr(owner, "rope_type", config.rope_parameters["rope_type"]))
    rope_init_fn = owner.compute_default_rope_parameters
    if rope_type != "default":
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
    inv_freq, attention_scaling = rope_init_fn(config, device)
    owner.attention_scaling = attention_scaling
    if name == "original_inv_freq":
        return inv_freq.clone()
    return inv_freq


register_rotary_materializer("Qwen2_5_VLRotaryEmbedding")(_materialize_rope_with_init_fn)
register_rotary_materializer("Qwen3RotaryEmbedding")(_materialize_rope_with_init_fn)


def _materialize_qwen_rotary_buffer(
    owner: nn.Module,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor | None:
    materializer = _ROTARY_MATERIALIZERS.get(type(owner).__name__)
    if materializer is not None:
        return materializer(owner, name, device)
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class NunchakuQwenCheckpointMetadata:
    config: dict[str, tp.Any]
    base_text_config: dict[str, tp.Any]
    base_vision_config: dict[str, tp.Any]
    quantization_config: dict[str, tp.Any]
    text_quantization_config: dict[str, tp.Any]
    vision_quantization_config: dict[str, tp.Any]
    text_encoder_usage: str


# ---------------------------------------------------------------------------
# P0 + P2: Unified base class with hook methods
# ---------------------------------------------------------------------------

class BaseNunchakuQwenEncoderModel(PreTrainedModel):
    """Unified base class for all Nunchaku Qwen encoder runtimes.

    Sub-classes only need to set a few class attributes and optionally
    override hook methods to customise config building, model construction,
    post-patch fixups, and validation.
    """

    base_model_prefix = "model"
    config_class: tp.ClassVar[tp.Any] = None
    _runtime_model_class: tp.ClassVar[tp.Any] = None
    _runtime_display_name: tp.ClassVar[str] = "Qwen encoder"

    # -- Hook: __init__ customisation ----------------------------------------

    def __init__(self, *, model: nn.Module, metadata: NunchakuQwenCheckpointMetadata) -> None:
        self._pre_super_init(model)
        super().__init__(model.config)
        self.model = model
        self.metadata = metadata
        self.config = model.config
        self._post_super_init(model)

    def _pre_super_init(self, model: nn.Module) -> None:
        """Called before ``PreTrainedModel.__init__``.  Default: set eager attn."""
        model.config._attn_implementation = "eager"

    def _post_super_init(self, model: nn.Module) -> None:
        """Called after ``PreTrainedModel.__init__``.  Default: no-op."""

    # -- Hook: dtype ----------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # -- Hook: config dict building (P2 validation lives here) ----------------

    @classmethod
    def _build_config_dict(cls, metadata: NunchakuQwenCheckpointMetadata) -> dict[str, tp.Any]:
        """Return the raw config dict used to construct the runtime config.

        Text-only encoders return ``metadata.base_text_config``;
        multimodal encoders override this to use ``build_full_qwen_config_dict``.
        """
        return dict(metadata.base_text_config)

    @classmethod
    def _validate_config(cls, config_dict: dict[str, tp.Any], metadata: NunchakuQwenCheckpointMetadata) -> None:
        """P2: Per-subclass config validation.  Default checks ``rope_scaling``."""
        rope_scaling = config_dict.get("rope_scaling")
        if rope_scaling is None:
            raise ValueError(
                "Missing `rope_scaling` in exported Qwen text metadata. "
                "Please re-export the text encoder with the updated deepcompressor exporter."
            )
        if not isinstance(rope_scaling, dict) or "mrope_section" not in rope_scaling:
            raise ValueError(
                "Invalid `rope_scaling` in exported Qwen text metadata. "
                "Expected a dict containing `mrope_section`."
            )

    # -- Hook: runtime config / model creation --------------------------------

    @classmethod
    def _build_runtime_config(cls, metadata: NunchakuQwenCheckpointMetadata, config_dict: dict[str, tp.Any]):
        if cls.config_class is None:
            raise TypeError(f"{cls.__name__} must define `config_class`.")
        config = cls.config_class(**config_dict)
        config._attn_implementation = "eager"
        return config

    @classmethod
    def _build_runtime_model(cls, config):
        if cls._runtime_model_class is None:
            raise TypeError(f"{cls.__name__} must define `_runtime_model_class`.")
        return cls._runtime_model_class(config)

    # -- Hook: post-patch fixup -----------------------------------------------

    @classmethod
    def _post_patch_config(cls, model: nn.Module, config_dict: dict[str, tp.Any]) -> None:
        """Fixup config attributes after quantized linears have been patched.

        Default: reset ``config._attn_implementation`` to ``"eager"``.
        Multimodal sub-classes override to set per-sub-model attn implementations.
        """
        model.config._attn_implementation = "eager"

    # -- Unified from_pretrained (P0) -----------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        pin_memory: bool | str = "auto",
    ):
        ckpt = resolve_checkpoint_path(pretrained_model_name_or_path)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        device = device if isinstance(device, torch.device) else torch.device(device)
        pin_memory_enabled = resolve_pin_memory(pin_memory, device)
        with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            metadata = load_qwen_checkpoint_metadata(handle)
            config_dict = cls._build_config_dict(metadata)
            cls._validate_config(config_dict, metadata)
            config = cls._build_runtime_config(metadata, config_dict)
            with init_empty_weights():
                model = cls._build_runtime_model(config)
            quantized_prefixes = collect_required_qwen_quantized_prefixes(
                handle,
                runtime_display_name=cls._runtime_display_name,
            )
            patch_qwen_runtime_linears(
                model,
                metadata=metadata,
                quantized_prefixes=quantized_prefixes,
                linear_dtype=torch_dtype,
            )
            cls._post_patch_config(model, config_dict)
            missing, unexpected = restore_qwen_runtime_model_tensors(
                handle,
                model=model,
                torch_dtype=torch_dtype,
                output_device=device,
                pin_memory=pin_memory_enabled,
            )
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to restore {cls._runtime_display_name} strictly enough. "
                f"missing={missing[:12]}, unexpected={unexpected[:12]}"
            )
        model.eval()
        return cls(model=model, metadata=metadata)

    # -- forward (sub-classes override) ---------------------------------------

    def forward(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")


# ---------------------------------------------------------------------------
# Checkpoint helpers (unchanged)
# ---------------------------------------------------------------------------

def _parse_json_metadata(metadata: dict[str, str], key: str, default: tp.Any) -> tp.Any:
    value = metadata.get(key)
    if value is None or not str(value).strip():
        return default
    return json.loads(value)


def resolve_checkpoint_path(pretrained_model_name_or_path: str | Path) -> Path:
    ckpt = Path(pretrained_model_name_or_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    return ckpt


def load_qwen_checkpoint_metadata(handle: safetensors.safe_open) -> NunchakuQwenCheckpointMetadata:
    metadata = dict(handle.metadata() or {})
    config = _parse_json_metadata(metadata, "config", {})
    base_text_config = _parse_json_metadata(metadata, "base_text_config", {})
    base_vision_config = _parse_json_metadata(metadata, "base_vision_config", {})
    quantization_config = _parse_json_metadata(metadata, "quantization_config", {})
    text_quantization_config = _parse_json_metadata(metadata, "text_quantization_config", quantization_config)
    vision_quantization_config = _parse_json_metadata(metadata, "vision_quantization_config", {})
    if not base_text_config and isinstance(config, dict):
        if isinstance(config.get("text_config"), dict):
            base_text_config = dict(config["text_config"])
        else:
            base_text_config = dict(config)
    if "rope_scaling" not in base_text_config and isinstance(config, dict) and config.get("rope_scaling") is not None:
        base_text_config = dict(base_text_config)
        base_text_config["rope_scaling"] = config["rope_scaling"]
    if not base_vision_config and isinstance(config, dict) and isinstance(config.get("vision_config"), dict):
        base_vision_config = dict(config["vision_config"])
    return NunchakuQwenCheckpointMetadata(
        config=tp.cast(dict[str, tp.Any], config if isinstance(config, dict) else {}),
        base_text_config=tp.cast(dict[str, tp.Any], base_text_config if isinstance(base_text_config, dict) else {}),
        base_vision_config=tp.cast(dict[str, tp.Any], base_vision_config if isinstance(base_vision_config, dict) else {}),
        quantization_config=tp.cast(dict[str, tp.Any], quantization_config if isinstance(quantization_config, dict) else {}),
        text_quantization_config=tp.cast(
            dict[str, tp.Any], text_quantization_config if isinstance(text_quantization_config, dict) else {}
        ),
        vision_quantization_config=tp.cast(
            dict[str, tp.Any], vision_quantization_config if isinstance(vision_quantization_config, dict) else {}
        ),
        text_encoder_usage=str(metadata.get("text_encoder_usage", "qwenimage_text_only")),
    )


def build_full_qwen_config_dict(metadata: NunchakuQwenCheckpointMetadata) -> dict[str, tp.Any]:
    config = dict(metadata.config)
    if not isinstance(config.get("text_config"), dict) and metadata.base_text_config:
        config["text_config"] = dict(metadata.base_text_config)
    elif isinstance(config.get("text_config"), dict) and metadata.base_text_config:
        merged_text_config = dict(config["text_config"])
        for key, value in metadata.base_text_config.items():
            merged_text_config.setdefault(key, value)
        config["text_config"] = merged_text_config
    if not isinstance(config.get("vision_config"), dict) and metadata.base_vision_config:
        config["vision_config"] = dict(metadata.base_vision_config)
    if "rope_scaling" not in config:
        if metadata.base_text_config.get("rope_scaling") is not None:
            config["rope_scaling"] = metadata.base_text_config["rope_scaling"]
        elif metadata.config.get("rope_scaling") is not None:
            config["rope_scaling"] = metadata.config["rope_scaling"]
    if isinstance(config.get("text_config"), dict) and "rope_scaling" not in config["text_config"] and config.get("rope_scaling") is not None:
        merged_text_config = dict(config["text_config"])
        merged_text_config["rope_scaling"] = config["rope_scaling"]
        config["text_config"] = merged_text_config
    return config


# ---------------------------------------------------------------------------
# Quantization helpers (unchanged)
# ---------------------------------------------------------------------------

def collect_qwen_quantized_prefixes(keys: tp.Iterable[str]) -> set[str]:
    return {key[: -len(".qweight")] for key in keys if key.endswith(".qweight")}


def collect_required_qwen_quantized_prefixes(
    handle: safetensors.safe_open, *, runtime_display_name: str
) -> set[str]:
    quantized_prefixes = collect_qwen_quantized_prefixes(handle.keys())
    if not quantized_prefixes:
        raise ValueError(
            f"Expected exported {runtime_display_name} runtime weights with quantized `qweight` tensors, "
            "but the checkpoint does not contain any quantized linear prefixes."
        )
    return quantized_prefixes


def _unpack_tinychat_w4(qweight: torch.Tensor, *, oc: int, ic: int) -> torch.Tensor:
    packed_oc = oc // 4
    if tuple(qweight.shape) != (packed_oc, ic):
        raise ValueError(f"Expected qweight shape {(packed_oc, ic)}, got {tuple(qweight.shape)}")
    packed = qweight.detach().cpu().to(dtype=torch.int16).contiguous()
    packed = packed.view(packed_oc, ic // 64, 4, 16).permute(0, 2, 1, 3).contiguous()
    packed_vals = packed.reshape(-1, 8).to(dtype=torch.int32)
    v0 = packed_vals & 0xF
    v1 = (packed_vals >> 4) & 0xF
    v2 = (packed_vals >> 8) & 0xF
    v3 = (packed_vals >> 12) & 0xF
    return torch.stack([v0, v1, v2, v3], dim=1).reshape(oc, ic)


def _dequant_tinychat_linear(
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    target_dtype: torch.dtype = torch.bfloat16,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if group_size <= 0:
        raise ValueError(f"Invalid group_size: {group_size}")
    oc = int(scales.shape[1])
    ic = int(qweight.shape[1])
    if oc % 4 != 0:
        raise ValueError(f"Expected output features divisible by 4, got {oc}")
    if ic % group_size != 0:
        raise ValueError(f"Expected input features {ic} divisible by group_size {group_size}")
    ng = ic // group_size
    q_int = _unpack_tinychat_w4(qweight, oc=oc, ic=ic).to(dtype=torch.float32)
    scale = scales[:ng].detach().cpu().transpose(0, 1).reshape(oc, ng, 1).to(dtype=torch.float32)
    zero = scaled_zeros[:ng].detach().cpu().transpose(0, 1).reshape(oc, ng, 1).to(dtype=torch.float32)
    weight = q_int.view(oc, ng, group_size) * scale + zero
    dequant_weight = weight.view(oc, ic).to(dtype=target_dtype)
    dequant_bias = None if bias is None else bias.detach().cpu().to(dtype=target_dtype)
    return dequant_weight, dequant_bias


def _resolve_quant_group_size(metadata: NunchakuQwenCheckpointMetadata) -> int:
    quant_configs = (
        metadata.quantization_config,
        metadata.text_quantization_config,
        metadata.vision_quantization_config,
    )
    for config in quant_configs:
        group_size = int(config.get("weight", {}).get("group_size", -1))
        if group_size > 0:
            return group_size
    return 128


def _resolve_qwen_runtime_group_size(metadata: NunchakuQwenCheckpointMetadata) -> int:
    group_size = _resolve_quant_group_size(metadata)
    if group_size <= 0:
        raise ValueError(f"Invalid Qwen runtime group size: {group_size}")
    return group_size


# ---------------------------------------------------------------------------
# Tensor transfer helpers (unchanged)
# ---------------------------------------------------------------------------

def _maybe_pin_tensor(tensor: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled or tensor.device.type != "cpu" or tensor.numel() == 0:
        return tensor
    try:
        return tensor if tensor.is_pinned() else tensor.pin_memory()
    except Exception:
        return tensor


def _move_tensor_to_device(tensor: torch.Tensor, *, device: torch.device | None, pin_memory: bool) -> torch.Tensor:
    if device is None or tensor.device == device:
        return tensor
    tensor = _maybe_pin_tensor(tensor, enabled=pin_memory and device.type == "cuda")
    non_blocking = bool(pin_memory and tensor.device.type == "cpu" and device.type == "cuda")
    return tensor.to(device=device, non_blocking=non_blocking)


def _move_and_cast_loaded_tensor(
    tensor: torch.Tensor,
    *,
    device: str | torch.device | None,
    dtype: torch.dtype | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    target_device = normalize_device(device) if device is not None else None
    if target_device is not None and tensor.device != target_device:
        tensor = _move_tensor_to_device(tensor, device=target_device, pin_memory=pin_memory)
    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _iter_module_tensor_owners(
    module: nn.Module,
) -> tp.Iterator[tuple[nn.Module, str, str, torch.Tensor | nn.Parameter]]:
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False)):
            yield submodule, "_parameters", name, param
        for name, buffer in list(submodule.named_buffers(recurse=False)):
            if buffer is None:
                continue
            yield submodule, "_buffers", name, buffer


def _replace_child_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, new_module)


def patch_qwen_runtime_linears(
    module: nn.Module,
    *,
    metadata: NunchakuQwenCheckpointMetadata,
    quantized_prefixes: set[str],
    linear_dtype: torch.dtype | None = None,
) -> None:
    group_size = _resolve_qwen_runtime_group_size(metadata)
    if not quantized_prefixes:
        return
    named_modules = dict(module.named_modules())
    missing: list[str] = []
    for module_name in sorted(quantized_prefixes):
        target = named_modules.get(module_name)
        if target is None:
            missing.append(module_name)
            continue
        if not isinstance(target, nn.Linear):
            raise TypeError(f"Expected `{module_name}` to be nn.Linear, got {type(target)}")
        if linear_dtype is not None and target.weight.dtype != linear_dtype:
            target.weight.data = target.weight.data.to(dtype=linear_dtype)
        qmodule = W4Linear.from_linear(target, group_size=group_size, init_only=True)
        _replace_child_module(module, module_name, qmodule)
    if missing:
        raise RuntimeError(
            "Checkpoint expects quantized Qwen linears that were not found in the runtime model: "
            f"{missing[:20]}"
        )


def materialize_meta_module_tensors(module: nn.Module, *, device: str | torch.device) -> None:
    target_device = normalize_device(device)
    for owner, store_name, name, tensor in _iter_module_tensor_owners(module):
        if tensor.device.type != "meta":
            continue
        if store_name == "_parameters":
            raise RuntimeError(
                f"Unexpected meta parameter `{name}` on {type(owner).__name__}; "
                "all parameters should come from the checkpoint."
            )
        replacement = _materialize_qwen_rotary_buffer(owner, name=name, device=target_device)
        if replacement is None:
            raise RuntimeError(
                f"Unexpected meta buffer `{name}` on {type(owner).__name__}; "
                "please add an explicit materialization rule instead of zero-initializing it."
            )
        if replacement.dtype != tensor.dtype:
            replacement = replacement.to(dtype=tensor.dtype)
        if tuple(replacement.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Materialized buffer `{name}` on {type(owner).__name__} with shape {tuple(replacement.shape)}, "
                f"expected {tuple(tensor.shape)}."
            )
        else:
            owner._buffers[name] = replacement


def load_qwen_runtime_state_dict(
    handle: safetensors.safe_open,
    *,
    torch_dtype: torch.dtype,
    output_device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for key in handle.keys():
        tensor = handle.get_tensor(key)
        state_dict[key] = _move_and_cast_loaded_tensor(
            tensor,
            device=output_device,
            dtype=torch_dtype if tensor.is_floating_point() else None,
            pin_memory=pin_memory,
        )
    return state_dict


def restore_qwen_runtime_model_tensors(
    handle: safetensors.safe_open,
    *,
    model: nn.Module,
    torch_dtype: torch.dtype,
    output_device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> tuple[list[str], list[str]]:
    state_dict = load_qwen_runtime_state_dict(
        handle,
        torch_dtype=torch_dtype,
        output_device=output_device,
        pin_memory=pin_memory,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    materialize_meta_module_tensors(model, device=output_device or "cpu")
    return missing, unexpected

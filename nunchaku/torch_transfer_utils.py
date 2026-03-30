from __future__ import annotations

import os
import platform
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from torch import nn


DEFAULT_PIPELINE_COMPONENT_ATTRS: tuple[str, ...] = ("text_encoder", "text_encoder_2", "vae", "unet", "transformer")
_PRETOUCHED_SIGNATURE_ATTR = "_nunchaku_pretouched_cpu_signature"
_PIPELINE_PRETOUCH_DECISIONS_ATTR = "_nunchaku_pretouch_auto_decisions"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_CpuTensorSignature = tuple[tuple[str, str, tuple[int, ...], str, int], ...]


def _env_flag(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def normalize_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _resolve_default_cuda_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    except Exception:
        return torch.device("cuda")


def _parse_bool_or_auto(value: bool | str, *, name: str) -> bool | Literal["auto"]:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported {name}={value!r}. Expected a boolean value or 'auto'.")


def _matches_default_h2d_staging_platform(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False

    if platform.machine().lower() != "aarch64":
        return False

    index = 0 if device.index is None else device.index
    try:
        if index < 0 or index >= torch.cuda.device_count():
            return False
        capability = torch.cuda.get_device_capability(index)
    except Exception:
        return False

    return capability[0] == 12


def _resolve_page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 4096


def _scan_module_cpu_tensors(module: nn.Module) -> tuple[list[torch.Tensor], _CpuTensorSignature]:
    tensors: list[torch.Tensor] = []
    signature: list[tuple[str, str, tuple[int, ...], str, int]] = []
    for submodule in module.modules():
        for name, tensor in list(submodule.named_parameters(recurse=False)) + list(submodule.named_buffers(recurse=False)):
            if tensor.device.type != "cpu" or tensor.numel() == 0:
                continue
            tensors.append(tensor)
            signature.append((type(tensor).__name__, name, tuple(tensor.shape), str(tensor.dtype), tensor.data_ptr()))
    return tensors, tuple(signature)


def _pretouch_tensor(tensor: torch.Tensor, *, page_size: int) -> None:
    storage = tensor.untyped_storage()
    storage_size = len(storage)
    if storage_size == 0:
        return

    checksum = 0
    for offset in range(0, storage_size, page_size):
        checksum += int(storage[offset])
    checksum += int(storage[storage_size - 1])
    _ = checksum


def _pretouch_module_cpu_tensors(module: nn.Module) -> tuple[int, bool]:
    tensors, signature = _scan_module_cpu_tensors(module)
    if not signature:
        return 0, False

    marker = signature
    if getattr(module, _PRETOUCHED_SIGNATURE_ATTR, None) == marker:
        return 0, False

    page_size = _resolve_page_size()
    with torch.no_grad():
        for tensor in tensors:
            _pretouch_tensor(tensor, page_size=page_size)

    setattr(module, _PRETOUCHED_SIGNATURE_ATTR, marker)
    return len(tensors), True


def _need_pretouch_static(device: torch.device) -> bool:
    override = _env_flag("NUNCHAKU_PRETOUCH_CPU_TENSORS")
    if override is None:
        override = _env_flag("NUNCHAKU_PRETOUCH_PIPELINE_CPU_TENSORS")
    if override in _TRUE_VALUES:
        return True
    if override in _FALSE_VALUES:
        return False

    return _matches_default_h2d_staging_platform(device)


def _need_pin_memory_static(device: torch.device) -> bool:
    override = _env_flag("NUNCHAKU_PIN_MEMORY")
    if override in _TRUE_VALUES:
        return True
    if override in _FALSE_VALUES:
        return False
    return _matches_default_h2d_staging_platform(device)


def resolve_pin_memory(pin_memory: bool | str, device: str | torch.device) -> bool:
    device = normalize_device(device)
    if device.type != "cuda":
        return False

    pin_memory = _parse_bool_or_auto(pin_memory, name="pin_memory")
    if pin_memory == "auto":
        return _need_pin_memory_static(device)
    return pin_memory


def pin_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sd.items():
        if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.numel() > 0:
            try:
                out[key] = value if value.is_pinned() else value.pin_memory()
            except Exception:
                out[key] = value
        else:
            out[key] = value
    return out


def resolve_pretouch_cpu_tensors(
    pretouch: bool | str,
    pipe: Any,
    device: str | torch.device,
    *,
    context: str = "pipeline_to_cuda",
    component_attrs: Sequence[str] = DEFAULT_PIPELINE_COMPONENT_ATTRS,
) -> bool:
    device = normalize_device(device)
    if device.type != "cuda":
        return False

    pretouch = _parse_bool_or_auto(pretouch, name="pretouch")
    if pretouch != "auto":
        return pretouch

    decisions = getattr(pipe, _PIPELINE_PRETOUCH_DECISIONS_ATTR, None)
    if not isinstance(decisions, dict):
        decisions = {}
        setattr(pipe, _PIPELINE_PRETOUCH_DECISIONS_ATTR, decisions)

    key = (
        device.type,
        device.index,
        context,
        tuple(component_attrs),
        _env_flag("NUNCHAKU_PRETOUCH_CPU_TENSORS"),
        _env_flag("NUNCHAKU_PRETOUCH_PIPELINE_CPU_TENSORS"),
    )
    if key in decisions:
        return decisions[key]

    decision = _need_pretouch_static(device)
    decisions[key] = decision
    return decision


def should_pretouch(device: str | torch.device | None = None) -> bool:
    """
    Return whether Nunchaku's default policy recommends pretouching CPU tensors.

    This is a public recommendation API. It does not trigger pretouch by
    itself; callers should use it to decide whether to invoke
    :func:`pretouch_pipeline_cpu_tensors` explicitly before ``pipe.to("cuda")``
    or before a CPU offload flow that will move model weights back to CUDA.

    When ``device`` is omitted, the current CUDA device is used if CUDA is
    available. This keeps the common single-GPU case concise while still
    allowing explicit multi-GPU selection.

    The default policy enables pretouch on ``aarch64`` systems with
    Blackwell-class CUDA GPUs (compute capability major version 12).
    Environment overrides via ``NUNCHAKU_PRETOUCH_CPU_TENSORS`` or
    ``NUNCHAKU_PRETOUCH_PIPELINE_CPU_TENSORS`` take precedence.
    """
    if device is None:
        resolved_device = _resolve_default_cuda_device()
        if resolved_device is None:
            return False
    else:
        resolved_device = normalize_device(device)
    return _need_pretouch_static(resolved_device)


def pretouch_pipeline_cpu_tensors(
    pipe: Any,
    component_attrs: Sequence[str] = DEFAULT_PIPELINE_COMPONENT_ATTRS,
    on_component: Callable[[str], None] | None = None,
) -> bool:
    """
    Pretouch common pipeline components on CPU before CUDA transfer.

    This is the main explicit execution API. It walks the known CPU-side
    pipeline components, touches one byte per memory page for each tensor,
    and skips components whose CPU tensor signature has not changed since the
    previous pretouch call.

    Returns ``True`` when at least one component was newly pretouched and
    ``False`` when nothing needed touching.
    """
    touched_any = False
    for attr in component_attrs:
        if not hasattr(pipe, attr):
            continue
        module = getattr(pipe, attr)
        if module is None or not isinstance(module, nn.Module):
            continue
        touched, did_pretouch = _pretouch_module_cpu_tensors(module)
        if on_component is not None and did_pretouch:
            on_component(attr)
        touched_any = touched > 0 or touched_any
    return touched_any


def maybe_pretouch_pipeline_cpu_tensors(
    pipe: Any,
    device: str | torch.device,
    *,
    pretouch: bool | str = "auto",
    context: str = "pipeline_to_cuda",
    component_attrs: Sequence[str] = DEFAULT_PIPELINE_COMPONENT_ATTRS,
    on_component: Callable[[str], None] | None = None,
) -> bool:
    """
    Conditionally pretouch pipeline CPU tensors using an explicit policy.

    This helper is still explicit: callers opt in by invoking it. When
    ``pretouch="auto"``, the decision follows :func:`should_pretouch` and the
    environment-variable overrides. When ``pretouch`` is a boolean, that value
    is used directly.
    """
    if not resolve_pretouch_cpu_tensors(
        pretouch,
        pipe,
        device,
        context=context,
        component_attrs=component_attrs,
    ):
        return False
    return pretouch_pipeline_cpu_tensors(
        pipe,
        component_attrs=component_attrs,
        on_component=on_component,
    )

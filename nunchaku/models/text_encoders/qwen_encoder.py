from __future__ import annotations

from pathlib import Path

import safetensors

from .qwen_common import load_qwen_checkpoint_metadata, resolve_checkpoint_path
from .qwen2_vl_edit_encoder import NunchakuQwen2VLEditEncoderModel
from .qwen2_vl_text_encoder import NunchakuQwen2VLTextEncoderModel

try:
    from .qwen3_text_encoder import NunchakuQwen3TextEncoderModel
except ImportError:  # pragma: no cover - optional on older transformers versions
    NunchakuQwen3TextEncoderModel = None  # type: ignore[assignment]

__all__ = ["NunchakuQwenEncoderModel"]


class NunchakuQwenEncoderModel:
    """Dispatch to the appropriate Qwen encoder runtime from one stable entry point."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        ckpt = resolve_checkpoint_path(pretrained_model_name_or_path)
        with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            metadata = load_qwen_checkpoint_metadata(handle)
            raw_metadata = dict(handle.metadata() or {})
        usage = metadata.text_encoder_usage.strip().lower()
        model_type = str(raw_metadata.get("model_type", "")).strip().lower()
        runtime_class = str(raw_metadata.get("runtime_class", "")).strip()
        if usage == "qwenimage_edit_multimodal" or model_type == "qwen2_vl_edit_encoder":
            return NunchakuQwen2VLEditEncoderModel.from_pretrained(ckpt, **kwargs)
        if usage == "qwen3_text_only" or model_type == "qwen3_text" or runtime_class == "NunchakuQwen3TextEncoderModel":
            if NunchakuQwen3TextEncoderModel is None:
                raise ImportError(
                    "Qwen3 runtime is not available in the installed environment. "
                    "Please upgrade transformers to a version that provides `Qwen3Model`."
                )
            return NunchakuQwen3TextEncoderModel.from_pretrained(ckpt, **kwargs)
        return NunchakuQwen2VLTextEncoderModel.from_pretrained(ckpt, **kwargs)

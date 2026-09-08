from __future__ import annotations

import typing as tp

import torch

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig as QwenTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel as QwenTextModel

from .qwen_common import BaseNunchakuQwenEncoderModel, NunchakuQwenCheckpointMetadata

__all__ = ["NunchakuQwen2VLTextEncoderModel"]

_TEXT_ONLY_UNSUPPORTED_ARGS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "rope_deltas",
    "mm_token_type_ids",
    "second_per_grid_ts",
    "labels",
    "logits_to_keep",
)


class NunchakuQwen2VLTextEncoderModel(BaseNunchakuQwenEncoderModel):
    config_class = QwenTextConfig
    _runtime_model_class = QwenTextModel
    _runtime_display_name = "Qwen text encoder"

    @classmethod
    def _validate_config(cls, config_dict: dict[str, tp.Any], metadata: NunchakuQwenCheckpointMetadata) -> None:
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

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: tp.Any,
    ) -> tp.Any:
        explicit_values = {
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "rope_deltas": rope_deltas,
            "mm_token_type_ids": mm_token_type_ids,
            "second_per_grid_ts": second_per_grid_ts,
            "labels": labels,
        }
        unsupported = [
            name
            for name, value in explicit_values.items()
            if name in _TEXT_ONLY_UNSUPPORTED_ARGS and value is not None
        ]
        if isinstance(logits_to_keep, torch.Tensor) or int(logits_to_keep) != 0:
            unsupported.append("logits_to_keep")
        if unsupported:
            raise NotImplementedError(
                f"{type(self).__name__} only supports the QwenImage text-only path; "
                f"unsupported kwargs: {sorted(unsupported)}"
            )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

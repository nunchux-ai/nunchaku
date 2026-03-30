from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from .qwen_common import (
    BaseNunchakuQwenEncoderModel,
    NunchakuQwenCheckpointMetadata,
    build_full_qwen_config_dict,
)

__all__ = ["NunchakuQwen2VLEditEncoderModel"]


class NunchakuQwen2VLEditEncoderModel(BaseNunchakuQwenEncoderModel):
    config_class = Qwen2_5_VLConfig
    _runtime_model_class = Qwen2_5_VLModel
    _runtime_display_name = "Qwen edit encoder"

    _EDIT_UNSUPPORTED_ARGS = ("labels", "logits_to_keep", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts")

    # -- __init__ hooks -------------------------------------------------------

    def _pre_super_init(self, model: nn.Module) -> None:
        # PreTrainedModel.__init__ may override _attn_implementation based on
        # config, so we temporarily set all sub-configs to "eager" and stash the
        # originals for post-init restoration.
        configs = self._collect_sub_configs(model)
        self._saved_attn_impl = {key: getattr(cfg, "_attn_implementation", None) for key, cfg in configs.items()}
        for cfg in configs.values():
            cfg._attn_implementation = "eager"

    def _post_super_init(self, model: nn.Module) -> None:
        for key, cfg in self._collect_sub_configs(self.model).items():
            original = self._saved_attn_impl.get(key)
            if original is not None:
                cfg._attn_implementation = original
        del self._saved_attn_impl

    # -- Config hooks ---------------------------------------------------------

    @classmethod
    def _build_config_dict(cls, metadata: NunchakuQwenCheckpointMetadata) -> dict[str, tp.Any]:
        return build_full_qwen_config_dict(metadata)

    @classmethod
    def _validate_config(cls, config_dict: dict[str, tp.Any], metadata: NunchakuQwenCheckpointMetadata) -> None:
        text_config = (
            config_dict.get("text_config")
            if isinstance(config_dict.get("text_config"), dict)
            else config_dict
        )
        rope_scaling = text_config.get("rope_scaling")
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

    @classmethod
    def _build_runtime_config(cls, metadata: NunchakuQwenCheckpointMetadata, config_dict: dict[str, tp.Any]):
        config = Qwen2_5_VLConfig(**config_dict)
        top_attn = cls._resolve_attn_implementation(config_dict, None)
        text_attn = cls._resolve_attn_implementation(config_dict, "text_config")
        vision_attn = cls._resolve_attn_implementation(config_dict, "vision_config")
        config._attn_implementation = top_attn
        config.text_config._attn_implementation = text_attn
        config.vision_config._attn_implementation = vision_attn
        return config

    @classmethod
    def _post_patch_config(cls, model: nn.Module, config_dict: dict[str, tp.Any]) -> None:
        model.config._attn_implementation = cls._resolve_attn_implementation(config_dict, None)
        model.language_model.config._attn_implementation = cls._resolve_attn_implementation(config_dict, "text_config")
        model.visual.config._attn_implementation = cls._resolve_attn_implementation(config_dict, "vision_config")

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _collect_sub_configs(model: nn.Module) -> dict[str, tp.Any]:
        configs: dict[str, tp.Any] = {"top": model.config}
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            configs["text"] = model.language_model.config
        if hasattr(model, "visual") and hasattr(model.visual, "config"):
            configs["vision"] = model.visual.config
        return configs

    @staticmethod
    def _resolve_attn_implementation(config_dict: dict[str, tp.Any], nested_key: str | None) -> str:
        if nested_key is None:
            value = config_dict.get("_attn_implementation")
        else:
            nested = config_dict.get(nested_key)
            value = nested.get("_attn_implementation") if isinstance(nested, dict) else None
            if value is None:
                value = config_dict.get("_attn_implementation")
        return str(value or "sdpa")

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
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
        unsupported: list[str] = []
        if labels is not None:
            unsupported.append("labels")
        if isinstance(logits_to_keep, torch.Tensor) or int(logits_to_keep) != 0:
            unsupported.append("logits_to_keep")
        if pixel_values_videos is not None:
            unsupported.append("pixel_values_videos")
        if video_grid_thw is not None:
            unsupported.append("video_grid_thw")
        if second_per_grid_ts is not None:
            unsupported.append("second_per_grid_ts")
        if unsupported:
            raise NotImplementedError(
                "NunchakuQwen2VLEditEncoderModel currently supports image editing only; "
                f"unsupported args: {sorted(unsupported)}"
            )
        if (pixel_values is not None or image_grid_thw is not None) and input_ids is None and inputs_embeds is None:
            raise ValueError("Multimodal edit encoding requires `input_ids` or `inputs_embeds` to place vision tokens.")
        if rope_deltas is not None:
            self.model.rope_deltas = rope_deltas
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            second_per_grid_ts=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

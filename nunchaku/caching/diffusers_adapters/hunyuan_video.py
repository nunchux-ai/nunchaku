"""Adapters for efficient caching in HunyuanVideo diffusion pipelines."""

import functools
from typing import Any, Dict, Optional, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from ..fbcache import (
    cache_context,
    check_and_apply_cache,
    create_cache_context,
    get_buffer,
    get_can_use_cache,
    set_buffer,
)


def apply_cache_on_pipe(pipe: DiffusionPipeline, **kwargs):
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context(create_cache_context()):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_cache_on_transformer(pipe.transformer, **kwargs)

    return pipe


def apply_cache_on_transformer(
    transformer,
    *,
    use_double_fb_cache: bool = False,
    residual_diff_threshold: float = 0.12,
    residual_diff_threshold_multi: float | None = None,
    residual_diff_threshold_single: float | None = None,
):
    if residual_diff_threshold_multi is None:
        residual_diff_threshold_multi = residual_diff_threshold

    if getattr(transformer, "_is_cached", False):
        transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
        transformer.residual_diff_threshold_single = residual_diff_threshold_single
        transformer.use_double_fb_cache = use_double_fb_cache
        return transformer

    transformer._original_forward = transformer.forward

    transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
    transformer.residual_diff_threshold_single = (
        residual_diff_threshold_single if residual_diff_threshold_single is not None else -1.0
    )
    transformer.use_double_fb_cache = use_double_fb_cache
    transformer.verbose = False

    transformer.forward = cached_forward_hunyuan_video.__get__(transformer, transformer.__class__)
    transformer._is_cached = True

    return transformer


def cached_forward_hunyuan_video(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[tuple[torch.Tensor], Transformer2DModelOutput]:
    if self.residual_diff_threshold_multi < 0.0:
        return self._original_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool)
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    block_args = (temb, attention_mask, image_rotary_emb, token_replace_emb, first_frame_num_tokens)

    original_hidden_states = hidden_states
    first_block = self.transformer_blocks[0]
    hidden_states, encoder_hidden_states = first_block(hidden_states, encoder_hidden_states, *block_args)
    first_residual_multi = hidden_states - original_hidden_states
    del original_hidden_states

    if self.use_double_fb_cache:
        call_remaining_fn = _run_remaining_multi_blocks
    else:
        call_remaining_fn = _run_remaining_all_blocks

    hidden_states, encoder_hidden_states, _ = check_and_apply_cache(
        first_residual=first_residual_multi,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        threshold=self.residual_diff_threshold_multi,
        parallelized=False,
        mode="multi",
        verbose=getattr(self, "verbose", False),
        call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: call_remaining_fn(
            self, hidden_states, encoder_hidden_states, *block_args
        ),
        remaining_kwargs={},
    )

    # Single-block caching is handled manually because HunyuanVideo's single blocks
    if self.use_double_fb_cache:
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        first_single_block = self.single_transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_single_block(hidden_states, encoder_hidden_states, *block_args)

        first_residual_single = hidden_states - original_hidden_states

        can_use_cache, diff = get_can_use_cache(
            first_residual_single, threshold=self.residual_diff_threshold_single, mode="single"
        )

        if can_use_cache:
            hs_residual = get_buffer("single_hidden_states_residual")
            enc_residual = get_buffer("single_encoder_hidden_states_residual")
            hidden_states = original_hidden_states + hs_residual
            encoder_hidden_states = original_encoder_hidden_states + enc_residual
            hidden_states = hidden_states.contiguous()
            encoder_hidden_states = encoder_hidden_states.contiguous()
        else:
            set_buffer("first_single_hidden_states_residual", first_residual_single)

            hidden_states, encoder_hidden_states, hs_residual, enc_residual = _run_remaining_single_blocks(
                self, hidden_states, encoder_hidden_states, *block_args
            )

            set_buffer("single_hidden_states_residual", hidden_states - original_hidden_states)
            set_buffer("single_encoder_hidden_states_residual", encoder_hidden_states - original_encoder_hidden_states)

        del original_hidden_states, original_encoder_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)


def _run_remaining_all_blocks(self, hidden_states, encoder_hidden_states, *block_args):
    """Run remaining dual-stream and all single-stream blocks (single-stage mode)."""
    original_h = hidden_states
    original_enc = encoder_hidden_states

    for block in self.transformer_blocks[1:]:
        hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *block_args)

    for block in self.single_transformer_blocks:
        hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *block_args)

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    hs_residual = hidden_states - original_h
    enc_residual = encoder_hidden_states - original_enc

    return hidden_states, encoder_hidden_states, hs_residual, enc_residual


def _run_remaining_multi_blocks(self, hidden_states, encoder_hidden_states, *block_args):
    """Run remaining dual-stream blocks only (double-stage mode, stage 1)."""
    original_h = hidden_states
    original_enc = encoder_hidden_states

    for block in self.transformer_blocks[1:]:
        hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *block_args)

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    hs_residual = hidden_states - original_h
    enc_residual = encoder_hidden_states - original_enc

    return hidden_states, encoder_hidden_states, hs_residual, enc_residual


def _run_remaining_single_blocks(self, hidden_states, encoder_hidden_states, *block_args):
    """Run remaining single-stream blocks (double-stage mode, stage 2)."""
    original_h = hidden_states
    original_enc = encoder_hidden_states

    for block in self.single_transformer_blocks[1:]:
        hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *block_args)

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    hs_residual = hidden_states - original_h
    enc_residual = encoder_hidden_states - original_enc

    return hidden_states, encoder_hidden_states, hs_residual, enc_residual

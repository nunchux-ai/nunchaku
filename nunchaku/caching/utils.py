"""
Caching utilities for transformer models.

Implements first-block caching to accelerate transformer inference by reusing computations
when input changes are minimal. Supports SANA and Flux architectures.

**Main Classes**

- :class:`SanaCachedTransformerBlocks` : Cached transformer blocks for SANA models.
- :class:`FluxCachedTransformerBlocks` : Cached transformer blocks for Flux models.

**Caching Strategy**

1. Compute the first transformer block.
2. Compare the residual with the cached residual.
3. If similar, reuse cached results for the remaining blocks; otherwise, recompute and update cache.

.. note::
   Adapted from ParaAttention:
   https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache/
"""

import torch
from torch import nn

from ..utils import pad_tensor
from .fbcache import apply_prev_hidden_states_residual, check_and_apply_cache, get_can_use_cache, set_buffer

num_transformer_blocks = 19  # FIXME
num_single_transformer_blocks = 38  # FIXME


class SanaCachedTransformerBlocks(nn.Module):
    """
    Caching wrapper for SANA transformer blocks.

    Parameters
    ----------
    transformer : nn.Module
        The original SANA transformer model to wrap.
    residual_diff_threshold : float
        Similarity threshold for cache validity.
    verbose : bool, optional
        Print caching status messages (default: False).

    Attributes
    ----------
    transformer : nn.Module
        Reference to the original transformer.
    transformer_blocks : nn.ModuleList
        The transformer blocks to cache.
    residual_diff_threshold : float
        Current similarity threshold.
    verbose : bool
        Verbosity flag.
    """

    def __init__(
        self,
        *,
        transformer=None,
        residual_diff_threshold,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.verbose = verbose

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        """
        Forward pass with caching for SANA transformer blocks.

        See also
        --------
        nunchaku.models.transformers.transformer_sana.NunchakuSanaTransformerBlocks.forward

        Notes
        -----
        If batch size > 2 or residual_diff_threshold <= 0, caching is disabled for now.
        """
        batch_size = hidden_states.shape[0]
        if self.residual_diff_threshold <= 0.0 or batch_size > 2:
            if batch_size > 2:
                print("Batch size > 2 (for SANA CFG) currently not supported")

            first_transformer_block = self.transformer_blocks[0]
            hidden_states = first_transformer_block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                height=post_patch_height,
                width=post_patch_width,
                skip_first_layer=False,
            )
            return hidden_states

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]

        hidden_states = first_transformer_block.forward_layer_at(
            0,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            height=post_patch_height,
            width=post_patch_width,
        )
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache, _ = get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
            parallelized=False,
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            if self.verbose:
                print("Cache hit!!!")
            hidden_states, _ = apply_prev_hidden_states_residual(hidden_states, None)
        else:
            if self.verbose:
                print("Cache miss!!!")
            set_buffer("first_multi_hidden_states_residual", first_hidden_states_residual)
            del first_hidden_states_residual

            hidden_states, hidden_states_residual = self.call_remaining_transformer_blocks(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
            )
            set_buffer("multi_hidden_states_residual", hidden_states_residual)
        torch._dynamo.graph_break()

        return hidden_states

    def call_remaining_transformer_blocks(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        """
        Call remaining SANA transformer blocks.

        Called when the cache is invalid. Skips the first layer and processes
        the remaining blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Hidden states from the first block.
        attention_mask : torch.Tensor
            Attention mask for the input.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        encoder_attention_mask : torch.Tensor, optional
            Encoder attention mask (default: None).
        timestep : torch.Tensor, optional
            Timestep tensor for conditioning (default: None).
        post_patch_height : int, optional
            Height after patch embedding (default: None).
        post_patch_width : int, optional
            Width after patch embedding (default: None).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Final hidden states after processing all blocks.
            - Residual difference for caching.
        """
        first_transformer_block = self.transformer_blocks[0]
        original_hidden_states = hidden_states
        hidden_states = first_transformer_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            height=post_patch_height,
            width=post_patch_width,
            skip_first_layer=True,
        )
        hidden_states_residual = hidden_states - original_hidden_states

        return hidden_states, hidden_states_residual


class FluxCachedTransformerBlocks(nn.Module):
    """
    Caching wrapper for Flux transformer blocks.

    Parameters
    ----------
    transformer : nn.Module
        The original Flux transformer model.
    use_double_fb_cache : bool, optional
        Cache both double and single transformer blocks (default: True).
    residual_diff_threshold_multi : float
        Similarity threshold for double blocks.
    residual_diff_threshold_single : float
        Similarity threshold for single blocks.
    return_hidden_states_first : bool, optional
        If True, return hidden states first (default: True).
    return_hidden_states_only : bool, optional
        If True, return only hidden states (default: False).
    verbose : bool, optional
        Print caching status messages (default: False).

    Attributes
    ----------
    transformer : nn.Module
        Reference to the original transformer.
    transformer_blocks : nn.ModuleList
        Double transformer blocks.
    single_transformer_blocks : nn.ModuleList
        Single transformer blocks.
    use_double_fb_cache : bool
        Whether both block types are cached.
    residual_diff_threshold_multi : float
        Threshold for double blocks.
    residual_diff_threshold_single : float
        Threshold for single blocks.
    return_hidden_states_first : bool
        Output order flag.
    return_hidden_states_only : bool
        Output type flag.
    verbose : bool
        Verbosity flag.
    m : object
        Nunchaku C model interface.
    dtype : torch.dtype
        Computation data type.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        *,
        transformer: nn.Module = None,
        use_double_fb_cache: bool = True,
        residual_diff_threshold_multi: float,
        residual_diff_threshold_single: float,
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        # self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.single_transformer_blocks = transformer.single_transformer_blocks

        self.use_double_fb_cache = use_double_fb_cache
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single

        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.verbose = verbose

        self.m = self.transformer_blocks[0].m
        self.dtype = torch.bfloat16 if self.m.isBF16() else torch.float16
        self.device = transformer.device

    @staticmethod
    def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
        """
        Packs rotary embeddings for efficient computation.

        See also
        --------
        nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformerBlocks.pack_rotemb
        """
        assert rotemb.dtype == torch.float32
        B = rotemb.shape[0]
        M = rotemb.shape[1]
        D = rotemb.shape[2] * 2
        msg_shape = "rotemb shape must be (B, M, D//2, 1, 2)"
        assert rotemb.shape == (B, M, D // 2, 1, 2), msg_shape
        assert M % 16 == 0
        assert D % 8 == 0
        rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
        rotemb = rotemb.permute(0, 1, 3, 2, 4)
        # 16*8 pack, FP32 accumulator (C) format
        # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
        rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
        rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
        rotemb = rotemb.contiguous()
        rotemb = rotemb.view(B, M, D)
        return rotemb

    def update_residual_diff_threshold(
        self, use_double_fb_cache=True, residual_diff_threshold_multi=0.12, residual_diff_threshold_single=0.09
    ):
        """
        Update caching configuration parameters.

        Parameters
        ----------
        use_double_fb_cache : bool, optional
            Use double first-block caching. Default is True.
        residual_diff_threshold_multi : float, optional
            Similarity threshold for Flux double blocks. Default is 0.12.
        residual_diff_threshold_single : float, optional
            Similarity threshold for Flux single blocks (used if
            ``use_double_fb_cache`` is False). Default is 0.09.

        Examples
        --------
        >>> cached_blocks.update_residual_diff_threshold(
        ...     use_double_fb_cache=False,
        ...     residual_diff_threshold_multi=0.15
        ... )
        """
        self.use_double_fb_cache = use_double_fb_cache
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
    ):
        """
        Forward pass with advanced caching for Flux transformer blocks.

        See also
        --------
        nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformerBlocks.forward

        Notes
        -----
        If batch size > 2 or residual_diff_threshold <= 0, caching is disabled for now.
        """
        # batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(original_device)
        temb = temb.to(self.dtype).to(original_device)
        image_rotary_emb = image_rotary_emb.to(original_device)

        if controlnet_block_samples is not None:
            controlnet_block_samples = (
                torch.stack(controlnet_block_samples).to(original_device) if len(controlnet_block_samples) > 0 else None
            )
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = (
                torch.stack(controlnet_single_block_samples).to(original_device)
                if len(controlnet_single_block_samples) > 0
                else None
            )

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        # [1, tokens, head_dim/2, 1, 2] (sincos)
        total_tokens = txt_tokens + img_tokens
        assert image_rotary_emb.shape[2] == 1 * total_tokens

        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = self.pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        if self.residual_diff_threshold_multi < 0.0:

            hidden_states = self.m.forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                rotary_emb_single,
                controlnet_block_samples,
                controlnet_single_block_samples,
                skip_first_layer,
            )

            hidden_states = hidden_states.to(original_dtype).to(original_device)

            encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
            hidden_states = hidden_states[:, txt_tokens:, ...]

            if self.return_hidden_states_only:
                return hidden_states
            if self.return_hidden_states_first:
                return hidden_states, encoder_hidden_states
            return encoder_hidden_states, hidden_states

        remaining_kwargs = {
            "temb": temb,
            "rotary_emb_img": rotary_emb_img,
            "rotary_emb_txt": rotary_emb_txt,
            "rotary_emb_single": rotary_emb_single,
            "controlnet_block_samples": controlnet_block_samples,
            "controlnet_single_block_samples": controlnet_single_block_samples,
            "txt_tokens": txt_tokens,
        }

        original_hidden_states = hidden_states
        first_hidden_states, first_encoder_hidden_states = self.m.forward_layer(
            0,
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            controlnet_block_samples,
            controlnet_single_block_samples,
        )
        hidden_states = first_hidden_states
        encoder_hidden_states = first_encoder_hidden_states
        first_hidden_states_residual_multi = hidden_states - original_hidden_states
        del original_hidden_states

        if self.use_double_fb_cache:
            call_remaining_fn = self.call_remaining_multi_transformer_blocks
        else:
            call_remaining_fn = self.call_remaining_FBCache_transformer_blocks

        torch._dynamo.graph_break()
        updated_h, updated_enc, threshold = check_and_apply_cache(
            first_residual=first_hidden_states_residual_multi,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            threshold=self.residual_diff_threshold_multi,
            parallelized=False,
            mode="multi",
            verbose=self.verbose,
            call_remaining_fn=call_remaining_fn,
            remaining_kwargs=remaining_kwargs,
        )
        self.residual_diff_threshold_multi = threshold
        if not self.use_double_fb_cache:
            if self.return_hidden_states_only:
                return updated_h
            if self.return_hidden_states_first:
                return updated_h, updated_enc
            return updated_enc, updated_h

        # DoubleFBCache
        cat_hidden_states = torch.cat([updated_enc, updated_h], dim=1)
        original_cat = cat_hidden_states
        cat_hidden_states = self.m.forward_single_layer(0, cat_hidden_states, temb, rotary_emb_single)

        first_hidden_states_residual_single = cat_hidden_states - original_cat
        del original_cat

        call_remaining_fn_single = self.call_remaining_single_transformer_blocks

        updated_cat, _, threshold = check_and_apply_cache(
            first_residual=first_hidden_states_residual_single,
            hidden_states=cat_hidden_states,
            encoder_hidden_states=None,
            threshold=self.residual_diff_threshold_single,
            parallelized=False,
            mode="single",
            verbose=self.verbose,
            call_remaining_fn=call_remaining_fn_single,
            remaining_kwargs=remaining_kwargs,
        )
        self.residual_diff_threshold_single = threshold

        # torch._dynamo.graph_break()

        final_enc = updated_cat[:, :txt_tokens, ...]
        final_h = updated_cat[:, txt_tokens:, ...]

        final_h = final_h.to(original_dtype).to(original_device)
        final_enc = final_enc.to(original_dtype).to(original_device)

        if self.return_hidden_states_only:
            return final_h
        if self.return_hidden_states_first:
            return final_h, final_enc
        return final_enc, final_h

    def call_remaining_FBCache_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=True,
        txt_tokens=None,
    ):
        """
        Call remaining Flux transformer blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        temb : torch.Tensor
            Time embedding tensor.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        rotary_emb_img : torch.Tensor
            Image rotary embeddings.
        rotary_emb_txt : torch.Tensor
            Text rotary embeddings.
        rotary_emb_single : torch.Tensor
            Single-head rotary embeddings.
        controlnet_block_samples : list, optional
            ControlNet block samples.
        controlnet_single_block_samples : list, optional
            ControlNet single block samples.
        skip_first_layer : bool, optional
            Whether to skip the first layer. Default is True.
        txt_tokens : int, optional
            Number of text tokens.

        Returns
        -------
        hidden_states : torch.Tensor
            Updated hidden states.
        encoder_hidden_states : torch.Tensor
            Updated encoder hidden states.
        hidden_states_residual : torch.Tensor
            Residual of hidden states.
        enc_residual : torch.Tensor
            Residual of encoder hidden states.
        """
        original_dtype = hidden_states.dtype
        original_device = hidden_states.device
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        hidden_states = self.m.forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_single,
            controlnet_block_samples,
            controlnet_single_block_samples,
            skip_first_layer,
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        enc_residual = encoder_hidden_states - original_encoder_hidden_states

        return hidden_states, encoder_hidden_states, hidden_states_residual, enc_residual

    def call_remaining_multi_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
        start_idx=1,
    ):
        """
        Call remaining Flux double blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        temb : torch.Tensor
            Time embedding tensor.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states.
        rotary_emb_img : torch.Tensor
            Image rotary embeddings.
        rotary_emb_txt : torch.Tensor
            Text rotary embeddings.
        rotary_emb_single : torch.Tensor
            Single-head rotary embeddings.
        controlnet_block_samples : list, optional
            ControlNet block samples.
        controlnet_single_block_samples : list, optional
            ControlNet single block samples.
        skip_first_layer : bool, optional
            Whether to skip the first layer. Default is False.
        txt_tokens : int, optional
            Number of text tokens.

        Returns
        -------
        hidden_states : torch.Tensor
            Updated hidden states.
        encoder_hidden_states : torch.Tensor
            Updated encoder hidden states.
        hidden_states_residual : torch.Tensor
            Residual of hidden states.
        enc_residual : torch.Tensor
            Residual of encoder hidden states.
        """
        original_hidden_states = hidden_states.clone()
        original_encoder_hidden_states = encoder_hidden_states.clone()

        for idx in range(start_idx, num_transformer_blocks):
            hidden_states, encoder_hidden_states = self.m.forward_layer(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                controlnet_block_samples,
                controlnet_single_block_samples,
            )

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hs_res = hidden_states - original_hidden_states
        enc_res = encoder_hidden_states - original_encoder_hidden_states
        return hidden_states, encoder_hidden_states, hs_res, enc_res

    def call_remaining_single_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
        start_idx=1,
    ):
        """
        Call remaining Flux single blocks.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states (concatenated).
        temb : torch.Tensor
            Time embedding tensor.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states (unused).
        rotary_emb_img : torch.Tensor
            Image rotary embeddings (unused).
        rotary_emb_txt : torch.Tensor
            Text rotary embeddings (unused).
        rotary_emb_single : torch.Tensor
            Single-head rotary embeddings.
        controlnet_block_samples : list, optional
            ControlNet block samples (unused).
        controlnet_single_block_samples : list, optional
            ControlNet single block samples (unused).
        skip_first_layer : bool, optional
            Whether to skip the first layer. Default is False.
        txt_tokens : int, optional
            Number of text tokens (unused).

        Returns
        -------
        hidden_states : torch.Tensor
            Updated hidden states.
        hidden_states_residual : torch.Tensor
            Residual of hidden states.
        """
        original_hidden_states = hidden_states.clone()

        for idx in range(start_idx, num_single_transformer_blocks):
            hidden_states = self.m.forward_single_layer(
                idx,
                hidden_states,
                temb,
                rotary_emb_single,
            )

        hidden_states = hidden_states.contiguous()
        hs_res = hidden_states - original_hidden_states
        return hidden_states, hs_res

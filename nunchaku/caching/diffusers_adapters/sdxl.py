"""
Adapters for efficient caching in SDXL diffusion pipelines.

This module enables first-block caching for SDXL models, providing:

- :func:`apply_cache_on_unet` — Add caching to a ``UNet2DConditionModel``.
- :func:`apply_cache_on_pipe` — Add caching to a complete SDXL pipeline.

Caching is context-managed and only active within a cache context.
"""

import functools

from diffusers import DiffusionPipeline

from .. import utils_sdxl
from ..fbcache import cache_context, create_cache_context, get_current_cache_context


def apply_cache_on_unet(unet, *, residual_diff_threshold=0.12, verbose=False):
    """
    Enable caching for a SDXL UNet2DConditionModel.

    This function wraps the UNet's forward method to use caching for faster inference.
    Uses single first-block caching with configurable similarity thresholds.

    Parameters
    ----------
    unet : UNet2DConditionModel
        The UNet to modify.
    residual_diff_threshold : float, optional
        Similarity threshold for caching (default: 0.12).
    verbose : bool, optional
        Print caching status messages (default: False).

    Returns
    -------
    UNet2DConditionModel
        The UNet with caching enabled.

    Notes
    -----
    If already cached, returns the UNet unchanged. Caching is only active within a cache context.
    """
    if getattr(unet, "_is_cached", False):
        return unet

    # Store original forward method
    unet._original_forward = unet.forward
    unet.residual_diff_threshold = residual_diff_threshold
    unet.verbose = verbose

    # Replace forward with cached version
    @functools.wraps(unet.forward)
    def new_forward(*args, **kwargs):
        cache_ctx = get_current_cache_context()
        if cache_ctx is not None:
            # Use cached forward
            return utils_sdxl.cached_forward_sdxl(unet, *args, **kwargs)
        else:
            # Use original forward
            return unet._original_forward(*args, **kwargs)

    unet.forward = new_forward
    unet._is_cached = True

    return unet


def apply_cache_on_pipe(pipe: DiffusionPipeline, *, residual_diff_threshold=0.12, verbose=False):
    """
    Enable caching for a complete SDXL diffusion pipeline.

    This function wraps the pipeline's ``__call__`` method to manage cache contexts,
    and applies UNet-level caching.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The SDXL pipeline to modify.
    residual_diff_threshold : float, optional
        Similarity threshold for caching (default: 0.12).
    verbose : bool, optional
        Print caching status messages (default: False).

    Returns
    -------
    DiffusionPipeline
        The pipeline with caching enabled.

    Notes
    -----
    Only this pipeline instance is patched. Other instances keep their original behavior.
    """
    # Wrap pipeline __call__ with cache context
    if not getattr(pipe, "_is_cached", False):
        original_class = pipe.__class__
        original_call = original_class.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context(create_cache_context()):
                return original_call(self, *args, **kwargs)

        # Special methods are resolved on the type; preserve the name used by save_pretrained.
        pipe.__class__ = type(
            original_class.__name__,
            (original_class,),
            {"__call__": new_call, "__module__": original_class.__module__},
        )
        pipe._is_cached = True

    # Apply caching to UNet
    apply_cache_on_unet(pipe.unet, residual_diff_threshold=residual_diff_threshold, verbose=verbose)

    return pipe

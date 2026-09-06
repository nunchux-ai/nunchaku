"""
Caching utilities for transformer models.

Implements first-block caching to accelerate transformer inference by reusing computations
when input changes are minimal. Supports SANA and Flux architectures.

**Main Classes**

- :class:`CacheContext` : Manages cache buffers and incremental naming.

**Key Functions**

- :func:`get_buffer`, :func:`set_buffer` : Cache buffer management.
- :func:`cache_context` : Context manager for cache operations.
- :func:`are_two_tensors_similar` : Tensor similarity check.
- :func:`apply_prev_hidden_states_residual` : Applies cached residuals.
- :func:`get_can_use_cache` : Checks cache usability.
- :func:`check_and_apply_cache` : Main cache logic.

**Caching Strategy**

1. Compute the first transformer block.
2. Compare the residual with the cached residual.
3. If similar, reuse cached results for the remaining blocks; otherwise, recompute and update cache.

.. note::
   Adapted from ParaAttention:
   https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache/
"""

import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Optional, Tuple

import torch


@dataclasses.dataclass
class CacheContext:
    """
    Manages cache buffers and incremental naming for transformer model inference.

    Attributes
    ----------
    buffers : Dict[str, torch.Tensor]
        Stores cached tensor buffers.
    incremental_name_counters : DefaultDict[str, int]
        Counters for generating unique incremental cache entry names.
    """

    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        """
        Generate an incremental cache entry name.

        Parameters
        ----------
        name : str, optional
            Base name for the counter. If None, uses "default".

        Returns
        -------
        str
            Incremental name in the format ``"{name}_{counter}"``.
        """
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_name(self):
        """
        Reset all incremental name counters.

        After calling this, :meth:`get_incremental_name` will start from 0 for each name.
        """
        self.incremental_name_counters.clear()

    # @torch.compiler.disable # This is a torchscript feature
    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
        """
        Retrieve a cached tensor buffer by name.

        Parameters
        ----------
        name : str
            Name of the buffer to retrieve.

        Returns
        -------
        torch.Tensor or None
            The cached tensor if found, otherwise None.
        """
        return self.buffers.get(name)

    def set_buffer(self, name: str, buffer: torch.Tensor):
        """
        Store a tensor buffer in the cache.

        Parameters
        ----------
        name : str
            The name to associate with the buffer.
        buffer : torch.Tensor
            The tensor to cache.
        """
        self.buffers[name] = buffer

    def clear_buffers(self):
        """
        Clear all cached tensor buffers.

        Removes all stored tensors from the cache.
        """
        self.buffers.clear()


@torch.compiler.disable
def get_buffer(name: str) -> torch.Tensor:
    """
    Retrieve a cached tensor buffer from the current cache context.

    Parameters
    ----------
    name : str
        The name of the buffer to retrieve.

    Returns
    -------
    torch.Tensor or None
        The cached tensor if found, otherwise None.

    Raises
    ------
    AssertionError
        If no cache context is currently active.

    Examples
    --------
    >>> with cache_context(create_cache_context()):
    ...     set_buffer("my_tensor", torch.randn(2, 3))
    ...     cached = get_buffer("my_tensor")
    """
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name: str, buffer: torch.Tensor):
    """
    Store a tensor buffer in the current cache context.

    Parameters
    ----------
    name : str
        The name to associate with the buffer.
    buffer : torch.Tensor
        The tensor to cache.

    Raises
    ------
    AssertionError
        If no cache context is currently active.

    Examples
    --------
    >>> with cache_context(create_cache_context()):
    ...     set_buffer("my_tensor", torch.randn(2, 3))
    ...     cached = get_buffer("my_tensor")
    """
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


_current_cache_context = None


def create_cache_context():
    """
    Create a new :class:`CacheContext` for managing cached computations.

    Returns
    -------
    CacheContext
        A new cache context instance.

    Examples
    --------
    >>> context = create_cache_context()
    >>> with cache_context(context):
    ...     # Cached operations here
    ...     pass
    """
    return CacheContext()


def get_current_cache_context():
    """
    Get the currently active cache context.

    Returns:
        CacheContext or None: The current cache context if one is active, None otherwise

    Example:
        >>> with cache_context(create_cache_context()):
        ...     current = get_current_cache_context()
        ...     assert current is not None
    """
    return _current_cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    """
    Context manager to set the active cache context.

    Sets the global cache context for the duration of the ``with`` block, restoring the previous context on exit.

    Parameters
    ----------
    cache_context : CacheContext
        The cache context to activate.

    Yields
    ------
    None

    Examples
    --------
    >>> context = create_cache_context()
    >>> with cache_context(context):
    ...     set_buffer("key", torch.tensor([1, 2, 3]))
    ...     cached = get_buffer("key")
    """
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable
def are_two_tensors_similar(t1: torch.Tensor, t2: torch.Tensor, *, threshold: float, parallelized: bool = False):
    """
    Check if two tensors are similar based on relative L1 distance.

    The relative distance is computed as
    ``mean(abs(t1 - t2)) / mean(abs(t1))`` and compared to ``threshold``.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor.
    t2 : torch.Tensor
        Second tensor.
    threshold : float
        Similarity threshold. Tensors are similar if relative distance < threshold.
    parallelized : bool, optional
        Unused. For API compatibility.

    Returns
    -------
    tuple of (bool, float)
        - bool: True if tensors are similar, False otherwise.
        - float: The computed relative L1 distance.
    """
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff_ratio = mean_diff / mean_t1

    is_similar = diff_ratio < threshold

    return is_similar, diff_ratio


@torch.compiler.disable
def apply_prev_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    mode: str = "multi",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply cached residuals to hidden states.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Current hidden states.
    encoder_hidden_states : torch.Tensor, optional
        Encoder hidden states (required for ``mode="multi"``).
    mode : {"multi", "single"}, default: "multi"
        Whether to apply residuals for Flux double blocks or single blocks.

    Returns
    -------
    tuple or torch.Tensor
        - If ``mode="multi"``: (updated_hidden_states, updated_encoder_hidden_states)
        - If ``mode="single"``: updated_hidden_states

    Raises
    ------
    AssertionError
        If required cached residuals are not found.
    ValueError
        If mode is not "multi" or "single".
    """
    if mode == "multi":
        hidden_states_residual = get_buffer("multi_hidden_states_residual")
        assert hidden_states_residual is not None, "multi_hidden_states_residual must be set before"
        hidden_states = hidden_states + hidden_states_residual
        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            enc_hidden_res = get_buffer("multi_encoder_hidden_states_residual")
            msg = "multi_encoder_hidden_states_residual must be set before"
            assert enc_hidden_res is not None, msg
            encoder_hidden_states = encoder_hidden_states + enc_hidden_res
            encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states

    elif mode == "single":
        single_residual = get_buffer("single_hidden_states_residual")
        msg = "single_hidden_states_residual must be set before"
        assert single_residual is not None, msg
        hidden_states = hidden_states + single_residual
        hidden_states = hidden_states.contiguous()

        return hidden_states

    else:
        raise ValueError(f"Unknown mode {mode}; expected 'multi' or 'single'")


@torch.compiler.disable
def get_can_use_cache(
    first_hidden_states_residual: torch.Tensor, threshold: float, parallelized: bool = False, mode: str = "multi"
):
    """
    Check if cached computations can be reused based on residual similarity.

    Parameters
    ----------
    first_hidden_states_residual : torch.Tensor
        Current first block residual.
    threshold : float
        Similarity threshold for cache validity.
    parallelized : bool, optional
        Whether computation is parallelized. Default is False.
    mode : {"multi", "single"}, optional
        Caching mode. Default is "multi".

    Returns
    -------
    tuple of (bool, float)
        - bool: True if cache can be used (residuals are similar), False otherwise.
        - float: The computed similarity difference, or threshold if no cache exists.

    Raises
    ------
    ValueError
        If mode is not "multi" or "single".
    """
    if mode == "multi":
        buffer_name = "first_multi_hidden_states_residual"
    elif mode == "single":
        buffer_name = "first_single_hidden_states_residual"
    else:
        raise ValueError(f"Unknown mode {mode}; expected 'multi' or 'single'")

    prev_res = get_buffer(buffer_name)

    if prev_res is None:
        return torch.tensor(False, device=first_hidden_states_residual.device), torch.tensor(
            threshold, device=first_hidden_states_residual.device
        )

    is_similar, diff = are_two_tensors_similar(
        prev_res,
        first_hidden_states_residual,
        threshold=threshold,
        parallelized=parallelized,
    )
    return is_similar, diff


def check_and_apply_cache(
    *,
    first_residual: torch.Tensor,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    threshold: float,
    parallelized: bool,
    mode: str,
    verbose: bool,
    call_remaining_fn,
    remaining_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """
    Check and apply cache based on residual similarity.

    This function determines whether cached results can be used by comparing the
    first block residuals. If the cache is valid, it applies cached computations;
    otherwise, it computes new values and updates the cache.

    Parameters
    ----------
    first_residual : torch.Tensor
        First block residual for similarity comparison.
    hidden_states : torch.Tensor
        Current hidden states.
    encoder_hidden_states : torch.Tensor, optional
        Encoder hidden states (required for "multi" mode).
    threshold : float
        Similarity threshold for cache validity.
    parallelized : bool
        Whether computation is parallelized.
    mode : {"multi", "single"}
        Caching mode.
    verbose : bool
        Whether to print caching status messages.
    call_remaining_fn : callable
        Function to call remaining transformer blocks.
    remaining_kwargs : dict
        Additional keyword arguments for `call_remaining_fn`.

    Returns
    -------
    tuple
        (updated_hidden_states, updated_encoder_hidden_states, threshold)
        - updated_hidden_states (torch.Tensor)
        - updated_encoder_hidden_states (torch.Tensor or None)
        - threshold (float)
    """
    can_use_cache, diff = get_can_use_cache(
        first_residual,
        threshold=threshold,
        parallelized=parallelized,
        mode=mode,
    )
    torch._dynamo.graph_break()

    if can_use_cache:
        if verbose:
            diff_val = diff.item() if isinstance(diff, torch.Tensor) else diff
            print(f"[{mode.upper()}] Cache hit! diff={diff_val:.4f}, " f"new threshold={threshold:.4f}")

        out = apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states, mode=mode)
        updated_h, updated_enc = out if isinstance(out, tuple) else (out, None)
        return updated_h, updated_enc, threshold

    old_threshold = threshold

    if verbose:
        diff_val = diff.item() if isinstance(diff, torch.Tensor) else diff
        print(f"[{mode.upper()}] Cache miss. diff={diff_val:.4f}, " f"was={old_threshold:.4f} => now={threshold:.4f}")

    if mode == "multi":
        set_buffer("first_multi_hidden_states_residual", first_residual)
    else:
        set_buffer("first_single_hidden_states_residual", first_residual)

    result = call_remaining_fn(
        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **remaining_kwargs
    )

    if mode == "multi":
        updated_h, updated_enc, hs_res, enc_res = result
        set_buffer("multi_hidden_states_residual", hs_res)
        set_buffer("multi_encoder_hidden_states_residual", enc_res)
        return updated_h, updated_enc, threshold

    elif mode == "single":
        updated_cat_states, cat_res = result
        set_buffer("single_hidden_states_residual", cat_res)
        return updated_cat_states, None, threshold

    raise ValueError(f"Unknown mode {mode}")

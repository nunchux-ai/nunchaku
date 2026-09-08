"""
Utility functions and classes for efficient transformer model management in Nunchaku.
"""

import copy

import torch
from torch import nn

from ..utils import copy_params_into


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    """
    Fuse a list of nn.Linear layers into a single nn.Linear with concatenated output features.

    Parameters
    ----------
    linears : list of nn.Linear
        List of linear layers to fuse. All must have the same input feature dimension.

    Returns
    -------
    fused : nn.Linear
        A new linear layer with concatenated output features and the same input features.

    Raises
    ------
    AssertionError
        If the input feature dimensions do not match.

    Notes
    -----
    The fused layer does not copy weights or biases from the input layers.
    """
    assert len(linears) > 0
    if len(linears) == 1:
        return linears[0]
    else:
        assert all(linear.in_features == linears[0].in_features for linear in linears)
        out_features = sum(linear.out_features for linear in linears)
        bias = all(linear.bias is not None for linear in linears)
        return nn.Linear(
            linears[0].in_features,
            out_features,
            bias=bias,
            dtype=linears[0].weight.dtype,
            device=linears[0].weight.device,
        )


class CPUOffloadManager:
    """
    Manager for per-transformer-block CPU offloading with asynchronous memory operations using a Ping-Pong buffer strategy.

    This class enables memory-efficient inference or training by keeping only a subset
    of transformer blocks on GPU, offloading the rest to CPU, and preloading blocks as needed.

    Parameters
    ----------
    blocks : list of nn.Module
        List of transformer blocks to manage.
    device : str or torch.device, optional
        Target CUDA device for GPU operations. Default is "cuda".
    use_pin_memory : bool, optional
        Whether to use pinned memory for faster CPU-to-GPU transfers. Default is True.
    on_gpu_modules : list of nn.Module, optional
        Additional modules to keep on GPU at all times. Default is [].
    num_blocks_on_gpu : int, optional
        Number of blocks to keep on GPU simultaneously. Must be > 0. Default is 1.
    empty_cache_freq : int, optional
        Frequency (in forward passes) to call torch.cuda.empty_cache(). Default is 0 (never).

    Attributes
    ----------
    blocks : list of nn.Module
        The managed transformer blocks.
    buffer_blocks : list of nn.Module
        Buffers for preloading blocks onto GPU.
    device : torch.device
        The current CUDA device.
    current_block_idx : int
        Index of the current block on GPU.
    forward_counter : int
        Number of forward passes completed.
    memory_stream : torch.cuda.Stream
        CUDA stream for memory operations.
    compute_done : torch.cuda.Event
        CUDA event signaling compute completion.
    memory_done : torch.cuda.Event
        CUDA event signaling memory completion.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: str | torch.device = torch.device("cuda"),
        use_pin_memory: bool = True,
        on_gpu_modules: list[nn.Module] = [],
        num_blocks_on_gpu: int = 1,
        empty_cache_freq: int = 0,
    ):
        self.blocks = blocks
        self.use_pin_memory = use_pin_memory
        self.on_gpu_modules = on_gpu_modules
        self.num_blocks_on_gpu = num_blocks_on_gpu
        assert self.num_blocks_on_gpu > 0

        # Two streams: one for compute, one for memory operations, will be initialized in set_device
        self.memory_stream = None

        self.compute_done = torch.cuda.Event(blocking=False)
        self.memory_done = torch.cuda.Event(blocking=False)

        # Build the ping-pong GPU scratch blocks. Deepcopy on CPU (not GPU) to
        # avoid a GPU-tensor deepcopy deadlock when this manager is constructed
        # inside a forward pass. set_device() relocates the blocks afterwards.
        first_param = next(blocks[0].parameters(), None)
        if first_param is not None and first_param.is_cuda:
            blocks[0].to("cpu")
        self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]

        self.device = None
        self.set_device(device)

        self.current_block_idx = 0
        self.forward_counter = 0
        self.empty_cache_freq = empty_cache_freq

    def set_device(self, device: torch.device | str, force: bool = False):
        """
        Set the CUDA device for offloading and memory operations.
        It will move buffer blocks and on-GPU modules to the specified device and offload other blocks to CPU, optionally using pinned memory.

        Parameters
        ----------
        device : torch.device or str
            Target CUDA device.
        force : bool, optional
            If True, force re-initialization even if device is unchanged. Default is False.

        Raises
        ------
        AssertionError
            If the device is not a CUDA device.
        """
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda"
        if self.device == device and not force:
            return
        self.device = device
        self.memory_stream = torch.cuda.Stream(device=device)
        for block in self.buffer_blocks:
            block.to(device)
        for module in self.on_gpu_modules:
            module.to(device)
        for i, block in enumerate(self.blocks):
            if i < self.num_blocks_on_gpu:
                block.to(device)
            else:
                block.to("cpu")
                if self.use_pin_memory:
                    for p in block.parameters(recurse=True):
                        p.data = p.data.pin_memory()
                    for b in block.buffers(recurse=True):
                        b.data = b.data.pin_memory()

    def load_block(self, block_idx: int, non_blocking: bool = True):
        """
        Move a transformer block from CPU to GPU buffer.

        Parameters
        ----------
        block_idx : int
            Index of the block to load.
        non_blocking : bool, optional
            Whether to use non-blocking memory copy. Default is True.

        Notes
        -----
        - No action is taken if the block is already on GPU or index is out of range.
        """
        # if the block is already on GPU, don't load it to the buffer
        if block_idx < self.num_blocks_on_gpu:
            return
        # if there are blocks on GPU, don't load the first block to the buffer again
        if block_idx >= len(self.blocks):
            return

        block = self.blocks[block_idx]
        copy_params_into(block, self.buffer_blocks[block_idx % 2], non_blocking=non_blocking)

    def step(self, compute_stream: torch.cuda.Stream | None = None):
        """
        Advance to the next transformer block, triggering asynchronous preloading.

        It will preload the next block onto GPU in the background and synchronize between compute and memory streams.
        After all the blocks are processed, it will call torch.cuda.empty_cache() periodically if ``empty_cache_freq`` > 0.

        Parameters
        ----------
        compute_stream : torch.cuda.Stream, optional
            CUDA stream for compute operations. If None, uses current stream.
        """
        if compute_stream is None:
            compute_stream = torch.cuda.current_stream()
        next_compute_done = torch.cuda.Event()
        next_compute_done.record(compute_stream)
        with torch.cuda.stream(self.memory_stream):
            self.memory_stream.wait_event(self.compute_done)
            self.load_block(self.current_block_idx + 1)  # if the current block is the last block, load the first block
            next_memory_done = torch.cuda.Event()
            next_memory_done.record(self.memory_stream)
        self.memory_done = next_memory_done
        self.compute_done = next_compute_done
        self.current_block_idx += 1
        if self.current_block_idx < len(self.blocks):
            # get ready for the next compute
            compute_stream.wait_event(self.memory_done)
        else:
            # ready to finish
            compute_stream.wait_event(self.compute_done)
            self.current_block_idx = 0
            self.forward_counter += 1
            if self.empty_cache_freq > 0 and self.forward_counter % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()

    def get_block(self, block_idx: int | None = None) -> nn.Module:
        """
        Retrieve the current or specified transformer block for computation.
        It will return a buffer block if the requested block is offloaded.

        Parameters
        ----------
        block_idx : int, optional
            Index of the block to retrieve. If None, returns the current block.

        Returns
        -------
        block : nn.Module
            The requested transformer block (on GPU if needed).
        """
        if block_idx is None:
            block_idx = self.current_block_idx
        if block_idx < self.num_blocks_on_gpu:
            return self.blocks[block_idx]
        else:
            return self.buffer_blocks[block_idx % 2]

    def initialize(self, stream: torch.cuda.Stream | None = None):
        """
        Initialize CUDA events for compute and memory streams.
        It will record the initial events for the compute and memory streams.

        Parameters
        ----------
        stream : torch.cuda.Stream, optional
            CUDA stream to record initial events. If None, uses current stream.

        Notes
        -----
        - Should be called before the first forward pass.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        self.compute_done.record(stream)
        self.memory_done.record(stream)

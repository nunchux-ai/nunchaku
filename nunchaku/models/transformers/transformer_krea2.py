"""
Nunchaku W4A4 runtime for the Krea 2 Turbo single-stream MMDiT.

Strategy (correctness-first MVP): subclass the diffusers ``Krea2Transformer2DModel`` and reuse its whole
forward pass (text fusion, [text, image] concat, 3D rotary, per-block AdaLN modulation, final layer)
unchanged. Only the linear projections inside the 28 ``Krea2TransformerBlock`` heavy blocks are replaced
with :class:`~nunchaku.models.linear.SVDQW4A4Linear`, which is where the SVDQuant W4A4 speedup lives. The
light per-token modules (img_in, time_embed, text_fusion, txt_in, final_layer) stay in bf16, matching what
deepcompressor skipped during calibration.

Follow-up optimizations (not needed to run): fuse q/k/v(+gate) into one GEMM with norm+rope applied inside
the kernel (see the Z-Image ``NunchakuZImageFusedModule``), and carry the gate as a 4th fused output.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    Krea2SwiGLU,
    Krea2Transformer2DModel,
    Krea2TransformerBlock,
)
from huggingface_hub import utils

from ...utils import get_precision
from ..attention import NunchakuBaseAttention
from ..attention_processors.krea2 import NunchakuKrea2AttnProcessor
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key


class NunchakuKrea2Attention(NunchakuBaseAttention):
    """W4A4 wrapper over :class:`~diffusers...Krea2Attention`.

    Keeps grouped-query metadata and the fp32 q/k RMSNorm modules from the source attention, and swaps the
    five projections (to_q, to_k, to_v, to_gate, to_out[0]) for :class:`SVDQW4A4Linear`.
    """

    def __init__(self, orig_attn: Krea2Attention, **kwargs):
        super().__init__("krea2")
        self.num_heads = orig_attn.num_heads
        self.num_kv_heads = orig_attn.num_kv_heads
        self.head_dim = orig_attn.head_dim
        self.hidden_size = orig_attn.hidden_size

        # Kept eager (fp32-normalized), unchanged from the reference.
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k

        # q/k/v/gate share one input, so the checkpoint calibrated them as a single group (one smooth,
        # one low-rank branch). Fuse into a single quantized GEMM; the processor splits the output.
        with torch.device("meta"):
            to_qkv_gate = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v, orig_attn.to_gate])
        self.to_qkv_gate = SVDQW4A4Linear.from_linear(to_qkv_gate, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

    def set_processor(self, processor: str):
        # Only one processor for Krea 2; the string is accepted for interface parity.
        self.processor = NunchakuKrea2AttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb)


class NunchakuKrea2SwiGLU(nn.Module):
    """W4A4 SwiGLU. Same forward as :class:`~diffusers...Krea2SwiGLU` with quantized gate/up/down."""

    def __init__(self, orig_ff: Krea2SwiGLU, **kwargs):
        super().__init__()
        self.gate = SVDQW4A4Linear.from_linear(orig_ff.gate, **kwargs)
        self.up = SVDQW4A4Linear.from_linear(orig_ff.up, **kwargs)
        self.down = SVDQW4A4Linear.from_linear(orig_ff.down, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class NunchakuKrea2Transformer2DModel(Krea2Transformer2DModel, NunchakuModelLoaderMixin):
    """Nunchaku-optimized Krea2Transformer2DModel. Inherits the diffusers forward; only the 28 heavy
    blocks' attention and SwiGLU are quantized."""

    def _patch_model(self, **kwargs):
        for block in self.transformer_blocks:
            assert isinstance(block, Krea2TransformerBlock)
            block.attn = NunchakuKrea2Attention(block.attn, **kwargs)
            block.ff = NunchakuKrea2SwiGLU(block.ff, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """Load a quantized Krea 2 transformer from a single Nunchaku-format safetensors file.

        The file carries the quantized weights for the 28 blocks plus a ``quantization_config`` metadata
        entry (rank, precision). Non-quantized modules (img_in, time_embed, text_fusion, txt_in,
        final_layer) are loaded in ``torch_dtype``.
        """
        device = kwargs.get("device", "cpu")
        if kwargs.get("offload", False):
            raise NotImplementedError("Offload is not supported for Krea2Transformer2DModel")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"

        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        # Metadata is authoritative when the converter wrote it; otherwise fall back to the
        # hardware default, as the other ports do. Silently defaulting to int4 would load an
        # nvfp4 checkpoint with the wrong kernels on Blackwell.
        precision = quantization_config.get("precision")
        if precision is None:
            precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        print(f"quantization_config: {quantization_config}, rank={rank}, precision={precision}")

        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)
        transformer.load_state_dict(model_state_dict)
        return transformer

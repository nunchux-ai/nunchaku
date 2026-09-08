"""
Implements the :class:`NunchakuFluxTransformer2dModel`, a quantized transformer for Diffusers with efficient inference and LoRA support.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from huggingface_hub import utils
from packaging.version import Version
from safetensors.torch import load_file
from torch import nn

from ..._C import QuantizedFluxModel
from ..._C import utils as cutils
from ...lora.flux.nunchaku_converter import fuse_vectors, to_nunchaku
from ...lora.flux.utils import is_nunchaku_format
from ...utils import check_hardware_compatibility, get_precision, load_state_dict_in_safetensors, pad_tensor
from .utils import NunchakuModelLoaderMixin

SVD_RANK = 32

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFluxTransformerBlocks(nn.Module):
    """
    Wrapper for quantized Nunchaku FLUX transformer blocks.

    This class manages the forward pass, rotary embedding packing, and optional
    residual callbacks for ID embeddings.

    Parameters
    ----------
    m : QuantizedFluxModel
        The quantized transformer model.
    device : str or torch.device
        Device to run the model on.
    """

    def __init__(self, m: QuantizedFluxModel, device: str | torch.device):
        super(NunchakuFluxTransformerBlocks, self).__init__()
        self.m = m
        self.dtype = torch.bfloat16 if m.isBF16() else torch.float16
        self.device = device

    @staticmethod
    def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
        """
        Packs rotary embeddings for efficient computation.

        Parameters
        ----------
        rotemb : torch.Tensor
            Rotary embedding tensor of shape (B, M, D//2, 1, 2), dtype float32.

        Returns
        -------
        torch.Tensor
            Packed rotary embedding tensor of shape (B, M, D).
        """
        assert rotemb.dtype == torch.float32
        B = rotemb.shape[0]
        M = rotemb.shape[1]
        D = rotemb.shape[2] * 2
        assert rotemb.shape == (B, M, D // 2, 1, 2)
        assert M % 16 == 0
        assert D % 8 == 0
        rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
        rotemb = rotemb.permute(0, 1, 3, 2, 4)
        # 16*8 pack, FP32 accumulator (C) format
        # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
        ##########################################|--M--|--D--|
        ##########################################|-3--4--5--6|
        ##########################################  :  :  :  :
        rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
        rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
        rotemb = rotemb.contiguous()
        rotemb = rotemb.view(B, M, D)
        return rotemb

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        id_embeddings=None,
        id_weight=None,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
    ):
        """
        Forward pass for the quantized transformer blocks.
        It will call the forward method of ``m`` on the C backend.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states for image tokens.
        temb : torch.Tensor
            Temporal embedding tensor.
        encoder_hidden_states : torch.Tensor
            Input hidden states for text tokens.
        image_rotary_emb : torch.Tensor
            Rotary embedding tensor for all tokens.
        id_embeddings : torch.Tensor, optional
            Optional ID embeddings for residual callback.
        id_weight : float, optional
            Weight for ID embedding residual.
        joint_attention_kwargs : dict, optional
            Additional kwargs for joint attention.
        controlnet_block_samples : list[torch.Tensor], optional
            ControlNet block samples.
        controlnet_single_block_samples : list[torch.Tensor], optional
            ControlNet single block samples.
        skip_first_layer : bool, optional
            Whether to skip the first layer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (encoder_hidden_states, hidden_states) after transformer blocks.
        """
        # batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        self.id_embeddings = id_embeddings
        self.id_weight = id_weight
        self.pulid_ca_idx = 0
        if self.id_embeddings is not None:
            self.set_pulid_residual_callback()

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)

        if controlnet_block_samples is not None:
            if len(controlnet_block_samples) > 0:
                controlnet_block_samples = torch.stack(controlnet_block_samples).to(self.device)
            else:
                controlnet_block_samples = None

        if controlnet_single_block_samples is not None:
            if len(controlnet_single_block_samples) > 0:
                controlnet_single_block_samples = torch.stack(controlnet_single_block_samples).to(self.device)
            else:
                controlnet_single_block_samples = None

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
        rotary_emb_single = image_rotary_emb  # .to(self.dtype)

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = self.pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))
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

        if self.id_embeddings is not None:
            self.reset_pulid_residual_callback()

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        return encoder_hidden_states, hidden_states

    def forward_layer_at(
        self,
        idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
    ):
        """
        Forward pass for a specific transformer layer in ``m``.

        Parameters
        ----------
        idx : int
            Index of the transformer layer.
        hidden_states : torch.Tensor
            Input hidden states for image tokens.
        encoder_hidden_states : torch.Tensor
            Input hidden states for text tokens.
        temb : torch.Tensor
            Temporal embedding tensor.
        image_rotary_emb : torch.Tensor
            Rotary embedding tensor for all tokens.
        joint_attention_kwargs : dict, optional
            Additional kwargs for joint attention.
        controlnet_block_samples : list[torch.Tensor], optional
            ControlNet block samples.
        controlnet_single_block_samples : list[torch.Tensor], optional
            ControlNet single block samples.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (encoder_hidden_states, hidden_states) after the specified layer.
        """
        # batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)
        temb = temb.to(self.dtype).to(self.device)
        image_rotary_emb = image_rotary_emb.to(self.device)

        if controlnet_block_samples is not None:
            controlnet_block_samples = torch.stack(controlnet_block_samples).to(self.device)
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = torch.stack(controlnet_single_block_samples).to(self.device)

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))

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

        hidden_states = hidden_states.to(original_dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(original_dtype).to(original_device)

        return encoder_hidden_states, hidden_states

    def set_pulid_residual_callback(self):
        """
        Sets the residual callback for PulID (personalized ID) embeddings.
        """
        id_embeddings = self.id_embeddings
        pulid_ca = self.pulid_ca
        pulid_ca_idx = [self.pulid_ca_idx]
        id_weight = self.id_weight

        def callback(hidden_states):
            ip = id_weight * pulid_ca[pulid_ca_idx[0]](id_embeddings, hidden_states)
            pulid_ca_idx[0] += 1
            return ip

        self.callback_holder = callback
        self.m.set_residual_callback(callback)

    def reset_pulid_residual_callback(self):
        """
        Resets the PulID residual callback to None.
        """
        self.callback_holder = None
        self.m.set_residual_callback(None)

    def __del__(self):
        """
        Destructor to reset the quantized model.
        """
        self.m.reset()

    def norm1(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the norm_one_forward for a specific layer in ``m``.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        emb : torch.Tensor
            Embedding tensor.
        idx : int, optional
            Layer index (default: 0).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Output tensors from norm_one_forward.
        """
        return self.m.norm_one_forward(idx, hidden_states, emb)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Rotary positional embedding function.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor of shape (..., n).
    dim : int
        Embedding dimension (must be even).
    theta : int
        Rotary base.

    Returns
    -------
    torch.Tensor
        Rotary embedding tensor.
    """
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    USE_SINCOS = True
    if USE_SINCOS:
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)
    else:
        out = out.view(batch_size, -1, dim // 2, 1, 1)

    return out.float()


class EmbedND(nn.Module):
    """
    Multi-dimensional rotary embedding module.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    theta : int
        Rotary base.
    axes_dim : list[int]
        List of axis dimensions for each spatial axis.
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super(EmbedND, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Computes rotary embeddings for multi-dimensional positions.

        Parameters
        ----------
        ids : torch.Tensor
            Position indices tensor of shape (..., n_axes).

        Returns
        -------
        torch.Tensor
            Rotary embedding tensor.
        """
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def load_quantized_module(
    path_or_state_dict: str | os.PathLike[str] | dict[str, torch.Tensor],
    device: str | torch.device = "cuda",
    use_fp4: bool = False,
    offload: bool = False,
    bf16: bool = True,
) -> QuantizedFluxModel:
    """
    Loads a quantized Nunchaku FLUX model from a state dict or file.

    Parameters
    ----------
    path_or_state_dict : str, os.PathLike, or dict
        Path to the quantized model file or a state dict.
    device : str or torch.device, optional
        Device to load the model on (default: "cuda").
    use_fp4 : bool, optional
        Whether to use FP4 quantization (default: False).
    offload : bool, optional
        Whether to offload weights to CPU (default: False).
    bf16 : bool, optional
        Whether to use bfloat16 (default: True).

    Returns
    -------
    QuantizedFluxModel
        Loaded quantized model.
    """
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedFluxModel()
    cutils.disable_memory_auto_release()
    m.init(use_fp4, offload, bf16, 0 if device.index is None else device.index)
    if isinstance(path_or_state_dict, dict):
        m.loadDict(path_or_state_dict, True)
    else:
        m.load(str(path_or_state_dict))
    return m


class NunchakuFluxTransformer2dModel(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku FLUX Transformer 2D Model.

    This class implements a quantized transformer model compatible with the Diffusers
    library, supporting LoRA, rotary embeddings, and efficient inference.

    Parameters
    ----------
    patch_size : int, optional
        Patch size for input images (default: 1).
    in_channels : int, optional
        Number of input channels (default: 64).
    out_channels : int or None, optional
        Number of output channels (default: None).
    num_layers : int, optional
        Number of transformer layers (default: 19).
    num_single_layers : int, optional
        Number of single transformer layers (default: 38).
    attention_head_dim : int, optional
        Dimension of each attention head (default: 128).
    num_attention_heads : int, optional
        Number of attention heads (default: 24).
    joint_attention_dim : int, optional
        Joint attention dimension (default: 4096).
    pooled_projection_dim : int, optional
        Pooled projection dimension (default: 768).
    guidance_embeds : bool, optional
        Whether to use guidance embeddings (default: False).
    axes_dims_rope : tuple[int], optional
        Axes dimensions for rotary embeddings (default: (16, 56, 56)).
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int] = (16, 56, 56),
    ):
        super(NunchakuFluxTransformer2dModel, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        # these state_dicts are used for supporting lora
        self._unquantized_part_sd: dict[str, torch.Tensor] = {}
        self._unquantized_part_loras: dict[str, torch.Tensor] = {}
        self._quantized_part_sd: dict[str, torch.Tensor] = {}
        self._quantized_part_vectors: dict[str, torch.Tensor] = {}
        self._original_in_channels = in_channels

        # ComfyUI LoRA related
        self.comfy_lora_meta_list = []
        self.comfy_lora_sd_list = []

        # Fast LoRA switching: preloaded variant pool
        self._lora_variants: dict[str, dict[str, torch.Tensor]] = {}
        self._lora_variant_unquantized: dict[str, dict[str, torch.Tensor]] = {}
        self._lora_variant_vectors: dict[str, dict[str, torch.Tensor]] = {}
        self._active_lora_variant: str | None = None

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Loads a Nunchaku FLUX transformer model from pretrained weights.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the model directory or HuggingFace repo.
        **kwargs
            Additional keyword arguments for device, offload, torch_dtype, precision, etc.

        Returns
        -------
        NunchakuFluxTransformer2dModel or (NunchakuFluxTransformer2dModel, dict)
            The loaded model, and optionally metadata if `return_metadata=True`.
        """
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        metadata = None

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        if pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ):
            transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
            quantized_part_sd = {}
            unquantized_part_sd = {}
            for k, v in model_state_dict.items():
                if k.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                    quantized_part_sd[k] = v
                else:
                    unquantized_part_sd[k] = v
            precision = get_precision(device=device)
            quantization_config = json.loads(metadata["quantization_config"])
            check_hardware_compatibility(quantization_config, device)
        else:
            transformer, unquantized_part_path, transformer_block_path = cls._build_model_legacy(
                pretrained_model_name_or_path, **kwargs
            )

            # get the default LoRA branch and all the vectors
            quantized_part_sd = load_file(transformer_block_path)
            unquantized_part_sd = load_file(unquantized_part_path)
        new_quantized_part_sd = {}
        for k, v in quantized_part_sd.items():
            if v.ndim == 1:
                new_quantized_part_sd[k] = v
            elif "qweight" in k:
                # only the shape information of this tensor is needed
                new_quantized_part_sd[k] = v.to("meta")

                # if the tensor has qweight, but does not have low-rank branch, we need to add some artificial tensors
                for t in ["lora_up", "lora_down"]:
                    new_k = k.replace(".qweight", f".{t}")
                    if new_k not in quantized_part_sd:
                        oc, ic = v.shape
                        ic = ic * 2  # v is packed into INT8, so we need to double the size
                        new_quantized_part_sd[k.replace(".qweight", f".{t}")] = torch.zeros(
                            (0, ic) if t == "lora_down" else (oc, 0), device=v.device, dtype=torch.bfloat16
                        )

            elif "lora" in k:
                new_quantized_part_sd[k] = v
        transformer._quantized_part_sd = new_quantized_part_sd
        m = load_quantized_module(
            quantized_part_sd,
            device=device,
            use_fp4=precision == "fp4",
            offload=offload,
            bf16=torch_dtype == torch.bfloat16,
        )
        transformer.inject_quantized_module(m, device)
        transformer.to_empty(device=device)

        transformer.load_state_dict(unquantized_part_sd, strict=False)
        transformer._unquantized_part_sd = unquantized_part_sd

        if kwargs.get("return_metadata", False):
            return transformer, metadata
        else:
            return transformer

    def inject_quantized_module(self, m: QuantizedFluxModel, device: str | torch.device = "cuda"):
        """
        Injects a quantized module into the model and sets up transformer blocks.

        Parameters
        ----------
        m : QuantizedFluxModel
            The quantized transformer model.
        device : str or torch.device, optional
            Device to run the model on (default: "cuda").

        Returns
        -------
        self : NunchakuFluxTransformer2dModel
            The model with injected quantized module.
        """
        print("Injecting quantized module")
        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])

        ### Compatible with the original forward method
        self.transformer_blocks = nn.ModuleList([NunchakuFluxTransformerBlocks(m, device)])
        self.single_transformer_blocks = nn.ModuleList([])

        return self

    def set_attention_impl(self, impl: str, attn_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Set the attention implementation for the quantized transformer block.

        Parameters
        ----------
        impl : str
            Attention implementation to use. Supported values:

            - ``"flashattn2"`` (default): Standard FlashAttention-2.
            - ``"nunchaku-fp16"``: Uses FP16 attention accumulation, up to 1.2× faster than FlashAttention-2 on NVIDIA 30-, 40-, and 50-series GPUs.
            - ``"custom"``: Custom attention implementation, requires `attn_func` to be provided.

        attn_func : Optional[Callable[[torch.Tensor], torch.Tensor]]
            Custom attention function to use if `impl` is set to `"custom"`.
            It should accept a single tensor `qkv` of shape `[batch, num_tokens, 3, num_heads, head_dim]` and return the attention output.
        """
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.setAttentionImpl(impl, attn_func or (lambda *args: None))

    ### LoRA Related Functions

    def _expand_module(self, module_name: str, new_shape: tuple[int, int]):
        """
        Expands a linear module to a new shape for LoRA compatibility.
        Mostly for FLUX.1-tools LoRA which changes the input channels.

        Parameters
        ----------
        module_name : str
            Name of the module to expand.
        new_shape : tuple[int, int]
            New shape (out_features, in_features) for the module.
        """
        module = self.get_submodule(module_name)
        assert isinstance(module, nn.Linear)
        weight_shape = module.weight.shape
        logger.info("Expand the shape of module {} from {} to {}".format(module_name, tuple(weight_shape), new_shape))
        assert new_shape[0] >= weight_shape[0] and new_shape[1] >= weight_shape[1]
        new_module = nn.Linear(
            new_shape[1],
            new_shape[0],
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        new_module.weight.data.zero_()
        new_module.weight.data[: weight_shape[0], : weight_shape[1]] = module.weight.data
        self._unquantized_part_sd[f"{module_name}.weight"] = new_module.weight.data.clone()
        if new_module.bias is not None:
            new_module.bias.data.zero_()
            new_module.bias.data[: weight_shape[0]] = module.bias.data
            self._unquantized_part_sd[f"{module_name}.bias"] = new_module.bias.data.clone()
        parent_name = ".".join(module_name.split(".")[:-1])
        parent_module = self.get_submodule(parent_name)
        parent_module.add_module(module_name.split(".")[-1], new_module)

        if module_name == "x_embedder":
            new_value = int(new_module.weight.data.shape[1])
            old_value = getattr(self.config, "in_channels")
            if new_value != old_value:
                logger.info(f"Update in_channels from {old_value} to {new_value}")
                setattr(self.config, "in_channels", new_value)

    def _update_unquantized_part_lora_params(self, strength: float = 1):
        """
        Updates the unquantized part of the model with LoRA parameters.

        Parameters
        ----------
        strength : float, optional
            LoRA scaling strength (default: 1).
        """
        # check if we need to expand the linear layers
        device = next(self.parameters()).device
        for k, v in self._unquantized_part_loras.items():
            if "lora_A" in k:
                lora_a = v
                lora_b = self._unquantized_part_loras[k.replace(".lora_A.", ".lora_B.")]
                diff_shape = (lora_b.shape[0], lora_a.shape[1])
                weight_shape = self._unquantized_part_sd[k.replace(".lora_A.", ".")].shape
                if diff_shape[0] > weight_shape[0] or diff_shape[1] > weight_shape[1]:
                    module_name = ".".join(k.split(".")[:-2])
                    self._expand_module(module_name, diff_shape)
            elif v.ndim == 1:
                diff_shape = v.shape
                weight_shape = self._unquantized_part_sd[k].shape
                if diff_shape[0] > weight_shape[0]:
                    assert diff_shape[0] >= weight_shape[0]
                    module_name = ".".join(k.split(".")[:-1])
                    module = self.get_submodule(module_name)
                    weight_shape = module.weight.shape
                    diff_shape = (diff_shape[0], weight_shape[1])
                    self._expand_module(module_name, diff_shape)
        new_state_dict = {}
        for k in self._unquantized_part_sd.keys():
            v = self._unquantized_part_sd[k]
            v = v.to(device)
            self._unquantized_part_sd[k] = v

            if v.ndim == 1 and k in self._unquantized_part_loras:
                diff = strength * self._unquantized_part_loras[k]
                if diff.shape[0] < v.shape[0]:
                    diff = torch.cat(
                        [diff, torch.zeros(v.shape[0] - diff.shape[0], device=device, dtype=v.dtype)], dim=0
                    )
                new_state_dict[k] = v + diff
            elif v.ndim == 2 and k.replace(".weight", ".lora_B.weight") in self._unquantized_part_loras:
                lora_a = self._unquantized_part_loras[k.replace(".weight", ".lora_A.weight")]
                lora_b = self._unquantized_part_loras[k.replace(".weight", ".lora_B.weight")]

                if lora_a.shape[1] < v.shape[1]:
                    lora_a = torch.cat(
                        [
                            lora_a,
                            torch.zeros(lora_a.shape[0], v.shape[1] - lora_a.shape[1], device=device, dtype=v.dtype),
                        ],
                        dim=1,
                    )
                if lora_b.shape[0] < v.shape[0]:
                    lora_b = torch.cat(
                        [
                            lora_b,
                            torch.zeros(v.shape[0] - lora_b.shape[0], lora_b.shape[1], device=device, dtype=v.dtype),
                        ],
                        dim=0,
                    )

                diff = strength * (lora_b @ lora_a)
                new_state_dict[k] = v + diff
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=True)

    def update_lora_params(self, path_or_state_dict: str | dict[str, torch.Tensor]):
        """
        Update the model with new LoRA parameters.

        Parameters
        ----------
        path_or_state_dict : str or dict
            Path to a LoRA weights file or a state dict. The path supports:

            - Local file path, e.g., ``"/path/to/your/lora.safetensors"``
            - HuggingFace repo with file, e.g., ``"user/repo/lora.safetensors"``
              (automatically downloaded and cached)
        """
        if isinstance(path_or_state_dict, dict):
            state_dict = {
                k: v for k, v in path_or_state_dict.items()
            }  # copy a new one to avoid modifying the original one
        else:
            state_dict = load_state_dict_in_safetensors(path_or_state_dict)

        if not is_nunchaku_format(state_dict):
            state_dict = to_nunchaku(state_dict, base_sd=self._quantized_part_sd)

        unquantized_part_loras = {}
        for k, v in list(state_dict.items()):
            device = next(self.parameters()).device
            if "transformer_blocks" not in k:
                unquantized_part_loras[k] = state_dict.pop(k).to(device)

        if len(self._unquantized_part_loras) > 0 or len(unquantized_part_loras) > 0:
            self._unquantized_part_loras = unquantized_part_loras

            self._unquantized_part_sd = {k: v for k, v in self._unquantized_part_sd.items() if "pulid_ca" not in k}
            self._update_unquantized_part_lora_params(1)

        quantized_part_vectors = {}
        for k, v in list(state_dict.items()):
            if v.ndim == 1:
                quantized_part_vectors[k] = state_dict.pop(k)
        if len(self._quantized_part_vectors) > 0 or len(quantized_part_vectors) > 0:
            self._quantized_part_vectors = quantized_part_vectors
            updated_vectors = fuse_vectors(quantized_part_vectors, self._quantized_part_sd, 1)
            state_dict.update(updated_vectors)

        # Get the vectors from the quantized part

        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)

        block.m.loadDict(state_dict, True)

    def set_lora_strength(self, strength: float = 1):
        """
        Sets the LoRA scaling strength for the model.

        Note: This function can only be used with a single LoRA. For multiple LoRAs,
        please fuse the LoRA scale into the weights.

        Parameters
        ----------
        strength : float, optional
            LoRA scaling strength (default: 1).

        Note: This function will change the strength of all the LoRAs. So only use it when you only have a single LoRA.
        """
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.setLoraScale(SVD_RANK, strength)
        if len(self._unquantized_part_loras) > 0:
            self._update_unquantized_part_lora_params(strength)
        if len(self._quantized_part_vectors) > 0:
            vector_dict = fuse_vectors(self._quantized_part_vectors, self._quantized_part_sd, strength)
            block.m.loadDict(vector_dict, True)

    def reset_x_embedder(self):
        """
        Resets the x_embedder module if the input channel count has changed.
        This is used for removing the effect of FLUX.1-tools LoRA which changes the input channels.
        """
        # if change the model in channels, we need to update the x_embedder
        if self._original_in_channels != self.config.in_channels:
            assert self._original_in_channels < self.config.in_channels
            old_module = self.x_embedder
            new_module = nn.Linear(
                in_features=self._original_in_channels,
                out_features=old_module.out_features,
                bias=old_module.bias is not None,
                device=old_module.weight.device,
                dtype=old_module.weight.dtype,
            )
            new_module.weight.data.copy_(old_module.weight.data[: new_module.out_features, : new_module.in_features])
            self._unquantized_part_sd["x_embedder.weight"] = new_module.weight.data.clone()
            if new_module.bias is not None:
                new_module.bias.data.zero_()
                new_module.bias.data.copy_(old_module.bias.data[: new_module.out_features])
                self._unquantized_part_sd["x_embedder.bias"] = new_module.bias.data.clone()
            self.x_embedder = new_module
            setattr(self.config, "in_channels", self._original_in_channels)

    def reset_lora(self):
        """
        Resets all LoRA parameters to their default state.
        """
        unquantized_part_loras = {}
        if len(self._unquantized_part_loras) > 0 or len(unquantized_part_loras) > 0:
            self._unquantized_part_loras = unquantized_part_loras
            self._update_unquantized_part_lora_params(1)
        state_dict = {k: v for k, v in self._quantized_part_sd.items() if "lora" in k}
        quantized_part_vectors = {}
        if len(self._quantized_part_vectors) > 0 or len(quantized_part_vectors) > 0:
            self._quantized_part_vectors = quantized_part_vectors
            updated_vectors = fuse_vectors(quantized_part_vectors, self._quantized_part_sd, 1)
            state_dict.update(updated_vectors)
        self.transformer_blocks[0].m.loadDict(state_dict, True)
        self.reset_x_embedder()

    def preload_loras(self, loras: dict[str, str | dict[str, torch.Tensor] | None]) -> None:
        """
        Preload multiple LoRAs for fast switching.

        This method pre-computes and caches the complete state dicts for each LoRA,
        enabling near-instant switching via :meth:`switch_lora`.

        Parameters
        ----------
        loras : dict
            Mapping of variant names to LoRA sources. Values can be:

            - ``str``: Path to a LoRA weights file (local or HuggingFace repo)
            - ``dict``: Pre-loaded state dict
            - ``None``: No LoRA (base model only)

        Example
        -------
        >>> transformer.preload_loras({
        ...     "none": None,
        ...     "anime": "path/to/anime.safetensors",
        ...     "realistic": "path/to/realistic.safetensors",
        ... })
        >>> transformer.switch_lora("anime")  # instant switch
        """
        device = next(self.parameters()).device

        for name, lora_source in loras.items():
            if lora_source is None:
                # Base model without LoRA
                quantized_state_dict = {k: v for k, v in self._quantized_part_sd.items() if "lora" in k}
                self._lora_variants[name] = {
                    k: v.to(device) if hasattr(v, "to") and v.device.type != "meta" else v
                    for k, v in quantized_state_dict.items()
                }
                self._lora_variant_unquantized[name] = {}
                self._lora_variant_vectors[name] = {}
            else:
                # Load and convert LoRA
                if isinstance(lora_source, dict):
                    state_dict = {k: v for k, v in lora_source.items()}
                else:
                    state_dict = load_state_dict_in_safetensors(lora_source)

                if not is_nunchaku_format(state_dict):
                    state_dict = to_nunchaku(state_dict, base_sd=self._quantized_part_sd)

                # Separate unquantized parts
                unquantized_loras = {}
                for k, v in list(state_dict.items()):
                    if "transformer_blocks" not in k:
                        unquantized_loras[k] = state_dict.pop(k).to(device)

                # Separate vector parts
                vectors = {}
                for k, v in list(state_dict.items()):
                    if v.ndim == 1:
                        vectors[k] = state_dict.pop(k)

                # Fuse vectors if present
                if vectors:
                    fused_vectors = fuse_vectors(vectors, self._quantized_part_sd, 1)
                    state_dict.update(fused_vectors)

                # Move to device and cache
                self._lora_variants[name] = {
                    k: v.to(device) if hasattr(v, "to") and v.device.type != "meta" else v
                    for k, v in state_dict.items()
                }
                self._lora_variant_unquantized[name] = unquantized_loras
                self._lora_variant_vectors[name] = vectors

        logger.info(f"Preloaded {len(loras)} LoRA variants: {list(loras.keys())}")

    def switch_lora(self, name: str, strength: float = 1.0) -> None:
        """
        Instantly switch to a preloaded LoRA variant.

        Parameters
        ----------
        name : str
            Name of the preloaded LoRA variant.
        strength : float, optional
            LoRA strength (default: 1.0).

        Raises
        ------
        KeyError
            If the variant name was not preloaded.

        Example
        -------
        >>> transformer.preload_loras({"anime": "path/to/anime.safetensors"})
        >>> transformer.switch_lora("anime")
        """
        if name not in self._lora_variants:
            available = list(self._lora_variants.keys())
            raise KeyError(f"LoRA variant '{name}' not preloaded. Available: {available}")

        # Update quantized parts (this is the fast path - just loadDict)
        block = self.transformer_blocks[0]
        assert isinstance(block, NunchakuFluxTransformerBlocks)
        block.m.loadDict(self._lora_variants[name], True)

        # Update unquantized parts
        self._unquantized_part_loras = self._lora_variant_unquantized[name]
        if self._unquantized_part_loras:
            self._update_unquantized_part_lora_params(strength)
        else:
            # Reset to base if no unquantized loras
            self._update_unquantized_part_lora_params(0)

        # Update vectors
        self._quantized_part_vectors = self._lora_variant_vectors[name]
        if self._quantized_part_vectors and strength != 1.0:
            vector_dict = fuse_vectors(self._quantized_part_vectors, self._quantized_part_sd, strength)
            block.m.loadDict(vector_dict, True)

        # Set strength
        block.m.setLoraScale(SVD_RANK, strength)

        self._active_lora_variant = name
        logger.debug(f"Switched to LoRA variant: {name}")

    def list_preloaded_loras(self) -> list[str]:
        """
        List all preloaded LoRA variant names.

        Returns
        -------
        list[str]
            Names of preloaded variants.
        """
        return list(self._lora_variants.keys())

    def get_active_lora(self) -> str | None:
        """
        Get the name of the currently active LoRA variant.

        Returns
        -------
        str or None
            Name of active variant, or None if not using preloaded variants.
        """
        return self._active_lora_variant

    def unload_lora_variant(self, name: str) -> None:
        """
        Unload a preloaded LoRA variant from GPU memory.

        Parameters
        ----------
        name : str
            Name of the variant to unload.

        Raises
        ------
        KeyError
            If the variant name was not preloaded.
        ValueError
            If trying to unload the currently active variant.
        """
        if name not in self._lora_variants:
            raise KeyError(f"LoRA variant '{name}' not preloaded.")
        if name == self._active_lora_variant:
            raise ValueError(f"Cannot unload active variant '{name}'. Switch to another first.")

        del self._lora_variants[name]
        del self._lora_variant_unquantized[name]
        del self._lora_variant_vectors[name]
        logger.info(f"Unloaded LoRA variant: {name}")

    def clear_preloaded_loras(self) -> None:
        """
        Clear all preloaded LoRA variants from GPU memory.

        Note: This also resets the model to base state (no LoRA).
        """
        self._lora_variants.clear()
        self._lora_variant_unquantized.clear()
        self._lora_variant_vectors.clear()
        self._active_lora_variant = None
        self.reset_lora()
        logger.info("Cleared all preloaded LoRA variants")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Forward pass for the Nunchaku FLUX transformer model.

        This method is compatible with the Diffusers pipeline and supports LoRA,
        rotary embeddings, and ControlNet.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            Input hidden states of shape (batch_size, channel, height, width).
        encoder_hidden_states : torch.FloatTensor, optional
            Conditional embeddings (e.g., prompt embeddings) of shape (batch_size, sequence_len, embed_dims).
        pooled_projections : torch.FloatTensor, optional
            Embeddings projected from the input conditions.
        timestep : torch.LongTensor, optional
            Denoising step.
        img_ids : torch.Tensor, optional
            Image token indices.
        txt_ids : torch.Tensor, optional
            Text token indices.
        guidance : torch.Tensor, optional
            Guidance tensor for classifier-free guidance.
        joint_attention_kwargs : dict, optional
            Additional kwargs for joint attention.
        controlnet_block_samples : list[torch.Tensor], optional
            ControlNet block samples.
        controlnet_single_block_samples : list[torch.Tensor], optional
            ControlNet single block samples.
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).
        controlnet_blocks_repeat : bool, optional
            Whether to repeat ControlNet blocks (default: False).

        Returns
        -------
        torch.FloatTensor or Transformer2DModelOutput
            Output tensor or output object containing the sample.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        nunchaku_block = self.transformer_blocks[0]
        encoder_hidden_states, hidden_states = nunchaku_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

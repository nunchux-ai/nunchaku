"""
Quantized linear layers for Nunchaku.
"""

import torch
from torch import nn

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda


class SVDQW4A4Linear(nn.Module):
    """
    `SVDQuant <paper_svdquant_>`_ W4A4 quantized linear layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    rank : int, optional
        SVD low-rank dimension. Default is 32.
    bias : bool, optional
        If True, adds a learnable bias. Default is True.
    precision : {'int4', 'nvfp4'}, optional
        Quantization precision data type ('int4' or 'nvfp4'). Default is 'int4'.
    act_unsigned : bool, optional
        If True, use unsigned activation quantization (int4 only). Default is False.
    torch_dtype : torch.dtype, optional
        Parameter dtype. Default is torch.bfloat16.
    device : str or torch.device or None, optional
        Device for parameters. Default is CPU.

    Attributes
    ----------
    in_features : int
    out_features : int
    rank : int
    precision : str
        'int4' or 'nvfp4'.
    group_size : int
        64 for int4, 16 for nvfp4.
    qweight : nn.Parameter
        Packed quantized weights, shape (out_features, in_features // 2), dtype int8.
    bias : nn.Parameter or None
        Bias tensor.
    wscales : nn.Parameter
        Weight scales, shape (in_features // group_size, out_features).
        Dtype: bfloat16/float16 (int4), float8_e4m3fn (nvfp4).
    smooth_factor : nn.Parameter
        Smoothing factors, shape (in_features,).
    smooth_factor_orig : nn.Parameter
        Original smoothing factors, shape (in_features,). (Unused)
    proj_down : nn.Parameter
        Packed low-rank down projection, shape (in_features, rank), dtype bfloat16/float16.
    proj_up : nn.Parameter
        Packed low-rank up projection, shape (out_features, rank), dtype bfloat16/float16.
    wtscale : float or None
        Global weight scale (nvfp4 only).
    wcscales : nn.Parameter or None
        Channel-wise weight scale (nvfp4 only), shape (out_features,), dtype float8_e4m3fn.
    act_unsigned : bool
        If True, input activations are unsigned (int4 only).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(SVDQW4A4Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.precision = precision
        self.torch_dtype = torch_dtype

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )

        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )

        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
            )
            self.wtscale = 1.0
        else:
            self.wtscale = None
            self.wcscales = None

        self.act_unsigned = act_unsigned

        # LoRA strength support (similar to Flux implementation)
        self.original_rank = rank  # Store original rank for LoRA scaling
        self.lora_strength = 1.0  # LoRA scaling strength

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        """
        Create an SVDQW4A4Linear from a standard nn.Linear. The weight and bias are dummy tensors.

        Parameters
        ----------
        linear : nn.Linear
            Source linear layer.
        **kwargs
            Additional init arguments.

        Returns
        -------
        SVDQW4A4Linear
        """
        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=linear.weight.device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with 16-bit input. It will call :meth:`quantize` and :meth:`forward_quant`.

        Parameters
        ----------
        x : torch.Tensor, shape (B, S, in_features), dtype float16 or bfloat16
            Input tensor.
        output : torch.Tensor or None, optional
            Optional output buffer.

        Returns
        -------
        torch.Tensor, shape (B, S, out_features)
            Output tensor.

        Notes
        -----
        B: batch size, S: sequence length
        """
        batch_size, seq_len, channels = x.shape
        x = x.reshape(batch_size * seq_len, channels)
        if output is None:
            output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        output = output.reshape(batch_size, seq_len, -1)
        return output

    def quantize(self, x: torch.Tensor, pad_size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input to 4-bit and compute low-rank hidden states. It will call :func:`~nunchaku.ops.quantize.svdq_quantize_w4a4_act_fuse_lora_cuda`.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_features), dtype float16 or bfloat16
            Input tensor.
        pad_size : int, optional
            Batch padding size. Default is 256.

        Returns
        -------
        quantized_x : torch.Tensor
            Quantized input, shape (pad_size * ceil(N / pad_size), in_features // 2), dtype uint8.
        ascales : torch.Tensor
            Activation scales, shape (in_features // group_size,), dtype float8_e4m3fn for nvfp4 and input dtype for int4.
        lora_act_out : torch.Tensor
            Low-rank hidden states, shape (pad_size * ceil(N / pad_size), rank), dtype float32.

        Notes
        -----
        N: batch size
        """
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x, lora_down=self.proj_down, smooth=self.smooth_factor, fp4=self.precision == "nvfp4", pad_size=pad_size
        )
        return quantized_x, ascales, lora_act_out

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-quantized input. It will call :func:`~nunchaku.ops.gemm.svdq_gemm_w4a4_cuda`.

        Parameters
        ----------
        quantized_x : torch.Tensor
            Quantized input, shape (N, in_features // 2), dtype uint8.
        ascales : torch.Tensor
            Activation scales, shape (in_features // group_size,), dtype float8_e4m3fn for nvfp4 and input dtype for int4.
        lora_act : torch.Tensor
            Low-rank hidden states, shape (N, rank), dtype float32.
        output : torch.Tensor or None, optional
            Optional output buffer.

        Returns
        -------
        torch.Tensor
            Output tensor, shape (N, out_features), dtype bfloat16/float16 for int4 and float8_e4m3fn for nvfp4.

        Notes
        -----
        N: batch size
        """
        if output is None:
            output = torch.empty(
                quantized_x.shape[0], self.out_features, dtype=self.proj_up.dtype, device=quantized_x.device
            )

        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
            lora_scales=self._get_lora_scales(),
        )
        return output

    def set_lora_strength(self, strength: float):
        """
        Set LoRA scaling strength for this layer.

        This method allows dynamic adjustment of LoRA strength, similar to Flux's setLoraScale.
        The strength is applied only to the LoRA part (ranks beyond original_rank), while
        the original low-rank branches remain at strength 1.0.

        For W4A4 quantized models, we apply an amplification factor to compensate for
        quantization precision loss, similar to how Flux handles this internally.

        Parameters
        ----------
        strength : float
            LoRA scaling strength. 1.0 means full LoRA effect, 0.0 means no LoRA effect.
        """
        # If strength is 0, disable LoRA effect completely
        if abs(strength) < 1e-6:
            self.lora_strength = 0.0
            print(
                f"[DEBUG] set_lora_strength: rank={self.rank}, original_rank={self.original_rank}, "
                f"input_strength={strength}, DISABLED (no LoRA effect)"
            )
            return

        # Apply amplification factor for W4A4 quantized models
        # This compensates for quantization precision loss
        AMPLIFICATION_FACTOR = 2.0  # Empirically determined factor for W4A4

        amplified_strength = strength * AMPLIFICATION_FACTOR
        self.lora_strength = amplified_strength

        print(
            f"[DEBUG] set_lora_strength: rank={self.rank}, original_rank={self.original_rank}, "
            f"input_strength={strength}, amplified_strength={amplified_strength}"
        )

    def _get_lora_scales(self) -> list[float]:
        """
        Get LoRA scaling factors for CUDA kernel.

        Returns
        -------
        list[float]
            Scaling factors for each 16-rank group.
            Original ranks (up to original_rank): 1.0
            LoRA ranks (beyond original_rank): lora_strength
        """
        import math

        # Calculate number of 16-rank groups
        total_groups = math.ceil(self.rank / 16)
        original_groups = math.ceil(self.original_rank / 16)

        # Build scales: original ranks = 1.0, LoRA ranks = lora_strength
        lora_scales = []
        for i in range(total_groups):
            if i < original_groups:
                lora_scales.append(1.0)  # Original low-rank branches
            else:
                lora_scales.append(self.lora_strength)  # LoRA weights

        # Debug output (disabled for cleaner logs)
        # print(f"[DEBUG] _get_lora_scales: rank={self.rank}, original_rank={self.original_rank}, "
        #       f"lora_strength={self.lora_strength}, scales={lora_scales}")

        return lora_scales

    def __repr__(self):
        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


class AWQW4A16Linear(nn.Module):
    """
    `AWQ <paper_awq_>`_ W4A16 quantized linear layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool, optional
        If True, adds learnable bias. Default is True.
    group_size : int, optional
        Quantization group size. Default is 64.
    torch_dtype : torch.dtype, optional
        Parameter dtype. Default is torch.bfloat16.
    device : str or torch.device or None, optional
        Device for parameters. Default is CPU.

    Attributes
    ----------
    in_features : int
    out_features : int
    group_size : int
    qweight : nn.Parameter
        Packed quantized weights, shape (out_features // 4, in_features // 2), dtype int32.
    bias : nn.Parameter or None
        Bias tensor.
    wscales : nn.Parameter
        Weight scales, shape (in_features // group_size, out_features), dtype float16 or bfloat16.
    wzeros : nn.Parameter
        Weight zero points, shape (in_features // group_size, out_features), dtype float16 or bfloat16.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(AWQW4A16Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(out_features // 4, in_features // 2, dtype=torch.int32, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AWQW4A16Linear.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_features)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (N, out_features)
            Output tensor.

        Notes
        -----
        N: batch size
        """
        output = awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self.qweight,
            scaling_factors=self.wscales,
            zeros=self.wzeros,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )
        if self.bias is not None:
            view_shape = [1] * (output.ndim - 1) + [-1]
            output.add_(self.bias.view(view_shape))
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Create an uninitialized AWQW4A16Linear from a standard nn.Linear.

        Parameters
        ----------
        linear : nn.Linear
            Source linear layer.
        group_size : int, optional
            Quantization group size.
        torch_dtype : torch.dtype, optional
            Parameter dtype.
        device : str, optional
            Device for parameters.

        Returns
        -------
        AWQW4A16Linear
        """
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def __repr__(self):
        return f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size})"

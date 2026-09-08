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
        )
        return output

    def __repr__(self):
        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )

    def update_lora_weights(
        self,
        extra_proj_down: torch.Tensor,
        extra_proj_up: torch.Tensor,
        strength: float = 1.0,
    ) -> None:
        """
        Update LoRA projection weights by fusing extra LoRA with existing SVD low-rank.

        Parameters
        ----------
        extra_proj_down : torch.Tensor
            Extra down-projection tensor, shape (extra_rank, in_features).
        extra_proj_up : torch.Tensor
            Extra up-projection tensor, shape (out_features, extra_rank).
        strength : float, optional
            LoRA strength multiplier. Default is 1.0.
        """
        from ..lora.flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight

        device = self.proj_down.device
        dtype = self.proj_down.dtype

        extra_proj_down = extra_proj_down.to(device=device, dtype=dtype)
        extra_proj_up = extra_proj_up.to(device=device, dtype=dtype) * strength

        unpacked_down = unpack_lowrank_weight(self.proj_down.data, down=True)
        unpacked_up = unpack_lowrank_weight(self.proj_up.data, down=False)

        fused_down = torch.cat([unpacked_down, extra_proj_down], dim=0)
        fused_up = torch.cat([unpacked_up, extra_proj_up], dim=1)

        packed_down = pack_lowrank_weight(fused_down, down=True)
        packed_up = pack_lowrank_weight(fused_up, down=False)

        new_rank = fused_down.shape[0]

        del self.proj_down
        del self.proj_up
        self.proj_down = nn.Parameter(packed_down.contiguous(), requires_grad=False)
        self.proj_up = nn.Parameter(packed_up.contiguous(), requires_grad=False)
        self.rank = new_rank

    def reset_lora_weights(
        self, original_proj_down: torch.Tensor, original_proj_up: torch.Tensor, original_rank: int
    ) -> None:
        """
        Reset LoRA weights to original state.

        Parameters
        ----------
        original_proj_down : torch.Tensor
            Original proj_down weights (packed format).
        original_proj_up : torch.Tensor
            Original proj_up weights (packed format).
        original_rank : int
            Original rank value.
        """
        device = self.proj_down.device
        dtype = self.proj_down.dtype

        del self.proj_down
        del self.proj_up
        self.proj_down = nn.Parameter(original_proj_down.to(device=device, dtype=dtype).contiguous(), requires_grad=False)
        self.proj_up = nn.Parameter(original_proj_up.to(device=device, dtype=dtype).contiguous(), requires_grad=False)
        self.rank = original_rank


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

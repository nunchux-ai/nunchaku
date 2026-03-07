__all__ = [
    "NunchakuFluxTransformer2dModel",
    "NunchakuChromaTransformer2dModel",
    "NunchakuSanaTransformer2DModel",
    "NunchakuT5EncoderModel",
    "NunchakuFluxTransformer2DModelV2",
    "NunchakuQwenImageTransformer2DModel",
    "NunchakuZImageTransformer2DModel",
]


def __getattr__(name: str):
    if name in __all__:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

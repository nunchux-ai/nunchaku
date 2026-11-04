import torch
from diffusers import ModularPipeline

from nunchaku.models.transformers.transformer_flux3 import NunchakuFlux3Transformer2DModel

if __name__ == "__main__":
    # Only an INT4 checkpoint is available, so this needs a GPU that runs INT4 models (e.g., RTX 30/40 series).
    transformer = NunchakuFlux3Transformer2DModel.from_pretrained(
        "Bluebear7878/nunchaku-flux-3-image-klein/svdq-int4_r32-flux-3-image-klein.safetensors"
    )

    blocks = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein").blocks
    # The default blocks rewrite the prompt with an 18.8 GB prompt upsampler; drop that step to encode the prompt as given.
    blocks.sub_blocks.pop("prompt_upsampler", None)
    pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein")
    # Load the text encoder and the VAE only.
    pipe.load_components(names=["text_encoder", "tokenizer", "vae"], dtype=torch.bfloat16)
    pipe.update_components(transformer=transformer)
    pipe.to("cuda")

    # FLUX.3 Image Klein expects a full scene caption; its model card recommends running the prompt upsampler first.
    prompt = (
        "A wide-angle photograph of a lone American bison in a misty geothermal landscape, half hidden by white steam, "
        "a slate-blue stream and evergreens fading into the mist behind. Muted, cool palette, soft overcast light."
    )
    image = pipe(prompt=prompt, height=1024, width=1024, generator=torch.Generator().manual_seed(0), output="images")[0]
    image.save("flux.3-image-klein-int4_r32.png")

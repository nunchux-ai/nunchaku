import gc
import os
from pathlib import Path

import diffusers
import pytest
import torch

from nunchaku.utils import get_precision, is_turing

from ...utils import already_generate, compute_lpips, hash_str_to_int

precision = get_precision()
torch_dtype = torch.bfloat16
dtype_str = "bf16"

model_name = "flux.3-image-klein"
width = 1024
height = 1024

ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
folder_name = f"w{width}h{height}"
save_dir_16bit = Path(ref_root) / model_name / dtype_str / folder_name

repo_id = "black-forest-labs/flux-3-image-klein"
nunchaku_repo_id = "NotaMG/flux-3-image-klein-svdq-int4-r32-nunchaku"

# FLUX.3 Image Klein expects full scene captions; its model card recommends running the prompt upsampler first.
dataset = [
    {
        "prompt": "Close-up outdoor photograph taken at the tiger’s eye level under diffused daylight, showing an adult Bengal tiger resting in tall grass. The tiger’s broad head and upper shoulders fill most of the frame, with its body lying low behind dense green and ochre blades. Its amber eyes are half closed, its black nose rests near the center, and its mouth is relaxed. One rounded ear sits partly hidden by grass while the other rises above the surrounding stems. Thick orange fur covers the face, patterned with irregular black stripes and pale cream markings around the eyes, cheeks, muzzle, and inner ears. The tiger’s left cheek presses into the grass, and several long blades cross in front of its muzzle and lower jaw. In the foreground, dry seed heads and narrow green leaves overlap the tiger’s chest, with one bent stalk crossing diagonally over its striped shoulder. The tiger’s forepaws are tucked beneath its body and mostly concealed by the grass. In the background, the vegetation continues as a soft mass of muted green and brown, with no open sky. A few seed heads catch the flat side light, while the tiger’s face remains sharply resolved against the softer surrounding grass.",
        "filename": "tiger",
    },
    {
        "prompt": 'Full-color outdoor photograph, straight-on view of a neighborhood bakery window under diffused morning daylight. Centered in the glass is a rectangular cream paper sign with uneven hand-painted edges and large handwritten black lettering reading "FRESH BREAD" in uppercase brush script. The sign is attached to the window with four small brass tacks, one near each corner, and the paper bows outward at the lower right corner. Behind the sign, the bakery interior is visible through the glass. On the left, a wooden shelf holds three round loaves with flour-dusted crusts and two long baguettes resting in a shallow wicker basket. On the right, a tiered wooden display contains braided bread, small rolls, and a row of scored sourdough loaves on a dark metal tray. Warm pendant lamps illuminate the bread and cast amber reflections across the window, while the handwritten sign remains clearly legible. Along the bottom of the display, a worn wooden counter holds a folded linen cloth and a pair of metal bread tongs. The window frame is painted muted green, with chipped paint around the lower sill. Reflections of pale building facades and a passing street cross the glass around the sign, including a blurred car shape near the left edge and a patch of overcast sky above. A small terracotta pot with a leafy herb sits on the sill at the far right.',
        "filename": "bakery_sign",
    },
    {
        "prompt": "Close-up indoor portrait photograph, shot at eye level in soft window light entering from the left. A young white woman in her twenties with fair skin and dense freckles across her cheeks, nose, and upper lip fills the center of the frame from the shoulders upward. She has shoulder-length wavy chestnut-brown hair parted off-center, with several loose strands resting against her temple. Her brown eyes look directly toward the camera, and her closed mouth rests in a neutral expression. She wears a faded sage-green linen blouse with a simple round neckline and a small silver stud earring. The window light spreads across the left side of her face, revealing the texture of her freckles and leaving the right side in a broad, pale shadow. A few flyaway hairs catch the light above her crown. Behind her, an off-white plaster wall and the edge of a tall window form a quiet background; the window frame is partly cropped on the left, with a sheer cream curtain diffusing the daylight. The background falls out of focus into muted beige and grey shapes, while the woman’s eyes, freckles, and the fine texture of her skin remain sharply resolved.",
        "filename": "portrait",
    },
]


def load_pipeline(transformer=None):
    blocks = diffusers.ModularPipeline.from_pretrained(repo_id).blocks
    # The default blocks rewrite the prompt with an 18.8 GB prompt upsampler, which full captions do not need.
    blocks.sub_blocks.pop("prompt_upsampler", None)
    pipe = blocks.init_pipeline(repo_id)
    names = ["text_encoder", "tokenizer", "vae"] + (["transformer"] if transformer is None else [])
    pipe.load_components(names=names, dtype=torch_dtype)
    if transformer is not None:
        pipe.update_components(transformer=transformer)
    return pipe.to("cuda")


def run_flux3_pipeline(pipe, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    for data in dataset:
        generator = torch.Generator().manual_seed(hash_str_to_int(data["filename"]))
        image = pipe(prompt=data["prompt"], height=height, width=width, generator=generator, output="images")[0]
        image.save(save_dir / f"{data['filename']}.png")


@pytest.mark.skipif(
    not hasattr(diffusers, "Flux3Transformer2DModel"), reason="diffusers has no Flux3Transformer2DModel"
)
@pytest.mark.skipif(is_turing(), reason="Turing GPUs. Skip tests.")
@pytest.mark.parametrize("rank,expected_lpips", [(32, {"int4-bf16": 0.25})])
def test_flux3_image_klein(rank: int, expected_lpips: dict[str, float]):
    if f"{precision}-{dtype_str}" not in expected_lpips:
        return

    if not already_generate(save_dir_16bit, len(dataset)):
        pipe = load_pipeline()
        run_flux3_pipeline(pipe, save_dir_16bit)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    from nunchaku.models.transformers.transformer_flux3 import NunchakuFlux3Transformer2DModel

    save_dir_nunchaku = (
        Path("test_results") / "nunchaku" / model_name / f"{precision}_r{rank}-{dtype_str}" / folder_name
    )
    path = f"{nunchaku_repo_id}/svdq-{precision}_r{rank}-flux3-image-klein.safetensors"
    transformer = NunchakuFlux3Transformer2DModel.from_pretrained(path, torch_dtype=torch_dtype)
    pipe = load_pipeline(transformer)
    run_flux3_pipeline(pipe, save_dir_nunchaku)
    del transformer
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    lpips = compute_lpips(save_dir_16bit, save_dir_nunchaku)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips[f"{precision}-{dtype_str}"] * 1.15

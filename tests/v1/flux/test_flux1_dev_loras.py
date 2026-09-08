import gc
import math
import os
from dataclasses import dataclass

import pytest
import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxAttention
from nunchaku.utils import get_precision, is_turing
from tests.data import get_dataset
from tests.flux.utils import offload_pipeline
from tests.utils import already_generate, compute_lpips, hash_str_to_int

LORA_PATH_MAP = {
    "hypersd8": "ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors",
    "turbo8": "alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors",
    "realism": "XLabs-AI/flux-RealismLora/lora.safetensors",
    "ghibsky": "aleksa-codes/flux-ghibsky-illustration/lora.safetensors",
    "anime": "alvdansen/sonny-anime-fixed/araminta_k_sonnyanime_fluxd_fixed.safetensors",
    "sketch": "Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch/FLUX-dev-lora-children-simple-sketch.safetensors",
    "yarn": "linoyts/yarn_art_Flux_LoRA/pytorch_lora_weights.safetensors",
    "haunted_linework": "alvdansen/haunted_linework_flux/hauntedlinework_flux_araminta_k.safetensors",
    "canny": "black-forest-labs/FLUX.1-Canny-dev-lora/flux1-canny-dev-lora.safetensors",
    "depth": "black-forest-labs/FLUX.1-Depth-dev-lora/flux1-depth-dev-lora.safetensors",
}


@dataclass
class Case:
    precision: str = "int4"
    dataset_name: str = "MJHQ"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 3.5
    attention_impl: str = "flashattn2"  # "flashattn2" or "nunchaku-fp16"
    lora_names: list[str] = None
    lora_strengths: list[float] = None
    expected_lpips: float = 0.5


def _run_test_case(test_case: Case, pytest_case_id: str):
    gc.collect()
    torch.cuda.empty_cache()

    model_name = "flux.1-dev"
    dtype = torch.bfloat16
    dtype_str = "bf16"
    max_dataset_size = 4

    dataset = get_dataset(name=test_case.dataset_name, max_dataset_size=max_dataset_size)

    model_id_16bit = "black-forest-labs/FLUX.1-dev"

    folder_name = f"w{test_case.width}h{test_case.height}t{test_case.num_inference_steps}g{test_case.guidance_scale}"
    assert test_case.lora_names is not None and len(test_case.lora_names) > 0
    assert test_case.lora_strengths is not None and len(test_case.lora_strengths) > 0
    assert len(test_case.lora_names) == len(test_case.lora_strengths)

    ref_root = os.environ.get("NUNCHAKU_TEST_CACHE_ROOT", os.path.join("test_results", "ref"))
    save_dir_16bit = os.path.join(ref_root, pytest_case_id, dtype_str, model_name, folder_name)

    # begin: run original flux
    if not already_generate(save_dir_16bit, max_dataset_size):
        pipeline = FluxPipeline.from_pretrained(model_id_16bit, torch_dtype=dtype)

        for i, lora_name in enumerate(test_case.lora_names):
            lora_path = LORA_PATH_MAP[lora_name]
            pipeline.load_lora_weights(
                os.path.dirname(lora_path), weight_name=os.path.basename(lora_path), adapter_name=f"lora_{i}"
            )
        pipeline.set_adapters([f"lora_{i}" for i in range(len(test_case.lora_strengths))], test_case.lora_strengths)
        pipeline = offload_pipeline(pipeline)
        _run_pipeline(
            dataset=dataset,
            pipeline=pipeline,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "height": test_case.height,
                "width": test_case.width,
                "num_inference_steps": test_case.num_inference_steps,
                "guidance_scale": test_case.guidance_scale,
            },
        )
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    # end: run original flux

    precision_str = test_case.precision
    if test_case.attention_impl == "flashattn2":
        precision_str += "-fa2"
    else:
        assert test_case.attention_impl == "nunchaku-fp16"
        precision_str += "-nfp16"

    save_dir_4bit = os.path.join("test_results", pytest_case_id, dtype_str, precision_str, model_name, folder_name)

    nunchaku_transformer: NunchakuFluxTransformer2DModelV2 = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{test_case.precision}_r32-flux.1-dev.safetensors", torch_dtype=dtype
    )
    for _, m in nunchaku_transformer.named_modules():
        if isinstance(m, NunchakuFluxAttention):
            m.set_processor(test_case.attention_impl)

    if len(test_case.lora_names) == 1:
        lora_path = LORA_PATH_MAP[test_case.lora_names[0]]
        lora_strength = test_case.lora_strengths[0]
        nunchaku_transformer.update_lora_params(lora_path)
        nunchaku_transformer.set_lora_strength(lora_strength)
    else:
        composed_lora = compose_lora(
            [
                (LORA_PATH_MAP[lora_name], lora_strength)
                for lora_name, lora_strength in zip(test_case.lora_names, test_case.lora_strengths)
            ]
        )
        nunchaku_transformer.update_lora_params(composed_lora)

    nunchaku_pipeline = FluxPipeline.from_pretrained(
        model_id_16bit, torch_dtype=dtype, transformer=nunchaku_transformer
    )
    nunchaku_pipeline = nunchaku_pipeline.to("cuda")

    _run_pipeline(
        dataset=dataset,
        pipeline=nunchaku_pipeline,
        save_dir=save_dir_4bit,
        forward_kwargs={
            "height": test_case.height,
            "width": test_case.width,
            "num_inference_steps": test_case.num_inference_steps,
            "guidance_scale": test_case.guidance_scale,
        },
    )
    del nunchaku_transformer
    del nunchaku_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    lpips = compute_lpips(save_dir_16bit, save_dir_4bit)
    print(f"lpips: {lpips}")
    assert lpips < test_case.expected_lpips * 1.15


def _run_pipeline(dataset, pipeline: FluxPipeline, save_dir: str, forward_kwargs: dict = {}):
    os.makedirs(save_dir, exist_ok=True)
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    for row in tqdm(
        dataset.iter(batch_size=1, drop_last_batch=False),
        desc="Batch",
        total=math.ceil(len(dataset)),
        position=0,
        leave=False,
    ):
        filenames = row["filename"]
        prompts = row["prompt"]
        seeds = [hash_str_to_int(filename) for filename in filenames]
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]
        images = pipeline(prompts, generator=generators, **forward_kwargs).images
        for i, image in enumerate(images):
            filename = filenames[i]
            image.save(os.path.join(save_dir, f"{filename}.png"))
        torch.cuda.empty_cache()


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    # "num_inference_steps,lora_name,lora_strength,cpu_offload,expected_lpips",
    # [
    #     (25, "realism", 0.9, True, 0.136 if get_precision() == "int4" else 0.112),
    #     # (25, "ghibsky", 1, False, 0.186),
    #     # (28, "anime", 1, False, 0.284),
    #     (24, "sketch", 1, True, 0.291 if get_precision() == "int4" else 0.221),
    #     # (28, "yarn", 1, False, 0.211),
    #     # (25, "haunted_linework", 1, True, 0.317),
    # ],
    "case",
    [
        pytest.param(
            Case(
                precision=get_precision(),
                dataset_name="realism",
                num_inference_steps=25,
                guidance_scale=3.5,
                attention_impl="nunchaku-fp16",
                lora_names=["realism"],
                lora_strengths=[0.9],
                expected_lpips=0.136 if get_precision() == "int4" else 0.112,
            ),
            id="flux.1-dev-r32-single-lora-realism",
        ),
        pytest.param(
            Case(
                precision=get_precision(),
                dataset_name="sketch",
                num_inference_steps=24,
                guidance_scale=3.5,
                attention_impl="flashattn2",
                lora_names=["sketch"],
                lora_strengths=[1.0],
                expected_lpips=0.291 if get_precision() == "int4" else 0.221,
            ),
            id="flux.1-dev-r32-single-lora-sketch",
        ),
    ],
)
def test_flux_dev_single_lora(case: Case, request):
    pytest_case_id = request.node.callspec.id
    _run_test_case(case, pytest_case_id)


# lora composition & large rank loras
@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            Case(
                precision=get_precision(),
                dataset_name="haunted_linework",
                num_inference_steps=8,
                guidance_scale=3.5,
                lora_names=["realism", "ghibsky", "anime", "sketch", "yarn", "haunted_linework", "turbo8"],
                lora_strengths=[0, 1, 0, 0, 0, 0, 1],
                expected_lpips=0.310 if get_precision() == "int4" else 0.217,
            ),
            id="flux.1-dev-r32-multi-loras-turbo8-ghibsky",
        )
    ],
)
def test_flux_dev_turbo8_ghibsky_1024x1024(case: Case, request):
    pytest_case_id = request.node.callspec.id
    _run_test_case(case, pytest_case_id)


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            Case(
                precision=get_precision(),
                dataset_name="haunted_linework",
                num_inference_steps=8,
                guidance_scale=3.5,
                lora_names=["realism", "ghibsky", "anime", "sketch", "yarn", "haunted_linework"],
                lora_strengths=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                expected_lpips=0.310 if get_precision() == "int4" else 0.217,
            ),
            id="flux.1-dev-r32-multi-loras",
        )
    ],
)
def test_flux_dev_multiple_loras(case: Case, request):
    pytest_case_id = request.node.callspec.id
    _run_test_case(case, pytest_case_id)


def test_kohya_lora():
    precision = get_precision()
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    transformer.update_lora_params("mit-han-lab/nunchaku-test-models/hand_drawn_game.safetensors")
    transformer.set_lora_strength(1)

    prompt = (
        "masterful impressionism oil painting titled 'the violinist', the composition follows the rule of thirds, "
        "placing the violinist centrally in the frame. the subject is a young woman with fair skin and light blonde "
        "hair is styled in a long, flowing hairstyle with natural waves. she is dressed in an opulent, "
        "luxurious silver silk gown with a high waist and intricate gold detailing along the bodice. "
        "the gown's texture is smooth and reflective. she holds a violin under her chin, "
        "her right hand poised to play, and her left hand supporting the neck of the instrument. "
        "she wears a delicate gold necklace with small, sparkling gemstones that catch the light. "
        "her beautiful eyes focused on the viewer. the background features an elegantly furnished room "
        "with classical late 19th century decor. to the left, there is a large, ornate portrait of "
        "a man in a dark suit, set in a gilded frame. below this, a wooden desk with a closed book. "
        "to the right, a red upholstered chair with a wooden frame is partially visible. "
        "the room is bathed in natural light streaming through a window with red curtains, "
        "creating a warm, inviting atmosphere. the lighting highlights the violinist, "
        "casting soft shadows that enhance the depth and realism of the scene, highly aesthetic, "
        "harmonious colors, impressioniststrokes, "
        "<lora:style-impressionist_strokes-flux-by_daalis:1.0> <lora:image_upgrade-flux-by_zeronwo7829:1.0>"
    )

    image = pipeline(prompt, num_inference_steps=20, guidance_scale=3.5).images[0]
    image.save(f"flux.1-dev-{precision}-1.png")

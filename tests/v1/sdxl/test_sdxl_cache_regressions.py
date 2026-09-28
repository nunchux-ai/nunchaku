import json

import pytest
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel

from nunchaku.caching.diffusers_adapters.sdxl import apply_cache_on_pipe, apply_cache_on_unet
from nunchaku.caching.fbcache import cache_context, create_cache_context, get_current_cache_context


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_unet(first_cross_attention=False):
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D" if first_cross_attention else "DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        cross_attention_dim=12,
        attention_head_dim=2,
        norm_num_groups=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=18,
    ).eval()


def unet_inputs():
    return {
        "sample": torch.randn(1, 4, 8, 8),
        "timestep": 1,
        "encoder_hidden_states": torch.randn(1, 2, 12),
        "added_cond_kwargs": {"text_embeds": torch.randn(1, 6), "time_ids": torch.randn(1, 6)},
    }


def make_pipeline():
    vae = AutoencoderKL(block_out_channels=(8,), norm_num_groups=4, latent_channels=4)
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        unet=make_unet(),
        scheduler=DDIMScheduler(),
        add_watermarker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def test_pipeline_cache_is_instance_local(monkeypatch):
    original_call = StableDiffusionXLPipeline.__call__
    # Restore the shared class even if a regression patches it before failing.
    monkeypatch.setattr(StableDiffusionXLPipeline, "__call__", original_call)
    monkeypatch.setattr(StableDiffusionXLPipeline, "_is_cached", False, raising=False)
    cached, uncached = make_pipeline(), make_pipeline()
    contexts = []
    hook = cached.unet.register_forward_pre_hook(lambda *_: contexts.append(get_current_cache_context()))
    apply_cache_on_pipe(cached, residual_diff_threshold=0)
    cached_class = type(cached)
    apply_cache_on_pipe(cached)

    assert type(cached) is cached_class
    assert StableDiffusionXLPipeline.__call__ is original_call
    assert type(uncached) is StableDiffusionXLPipeline
    assert not uncached._is_cached
    assert not getattr(uncached.unet, "_is_cached", False)
    assert isinstance(cached, StableDiffusionXLPipeline)
    assert cached.components["unet"] is cached.unet
    assert json.loads(cached.to_json_string())["_class_name"] == StableDiffusionXLPipeline.__name__

    kwargs = {
        "prompt_embeds": torch.randn(1, 2, 12),
        "pooled_prompt_embeds": torch.randn(1, 6),
        "latents": torch.randn(1, 4, 8, 8),
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "height": 8,
        "width": 8,
        "output_type": "latent",
    }
    with torch.no_grad():
        expected = uncached(**kwargs).images
        actual = cached(**kwargs).images
        repeated = cached(**kwargs).images
    hook.remove()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)
    assert contexts[0] is contexts[1]
    assert contexts[2] is contexts[3]
    assert contexts[0] is not contexts[2]
    assert get_current_cache_context() is None
    outer_context = create_cache_context()
    with cache_context(outer_context):
        with pytest.raises(ValueError, match="divisible by 8"):
            cached(**dict(kwargs, height=7))
        assert get_current_cache_context() is outer_context


@pytest.mark.parametrize("first_cross_attention", [False, True])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("legacy", [False, True])
def test_adapter_residuals_match_uncached_without_mutating_inputs(first_cross_attention, container, legacy):
    unet = make_unet(first_cross_attention)
    kwargs = unet_inputs()
    first_size = 8 if first_cross_attention else 4
    residuals = container(
        [torch.randn(1, 8, first_size, first_size), torch.randn(1, 16, 4, 4), torch.randn(1, 16, 4, 4)]
    )
    originals = tuple(residuals)
    with torch.no_grad():
        expected = unet(**kwargs, down_intrablock_additional_residuals=list(residuals)).sample
        apply_cache_on_unet(unet, residual_diff_threshold=0)
        key = "down_block_additional_residuals" if legacy else "down_intrablock_additional_residuals"
        with cache_context(create_cache_context()):
            actual = unet(**kwargs, **{key: residuals}).sample
    torch.testing.assert_close(actual, expected)
    assert len(residuals) == len(originals)
    assert all(actual is original for actual, original in zip(residuals, originals))


@pytest.mark.parametrize("controlnet", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_cache_miss_matches_uncached(controlnet, return_dict):
    unet = make_unet()
    kwargs = unet_inputs()
    kwargs["return_dict"] = return_dict
    if controlnet:
        kwargs["down_block_additional_residuals"] = tuple(
            torch.randn(shape) for shape in [(1, 8, 8, 8), (1, 8, 8, 8), (1, 8, 4, 4), (1, 16, 4, 4)]
        )
        kwargs["mid_block_additional_residual"] = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        expected = unet(**kwargs)[0]
        apply_cache_on_unet(unet, residual_diff_threshold=0)
        with cache_context(create_cache_context()):
            actual = unet(**kwargs)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cache_state", ["ready", "missing", "wrong_dtype"])
def test_cache_hit_requires_ready_output(cache_state):
    unet = make_unet()
    kwargs = unet_inputs()
    apply_cache_on_unet(unet, residual_diff_threshold=1.0)
    later_blocks = []
    hook = unet.down_blocks[1].register_forward_pre_hook(lambda *_: later_blocks.append(True))
    context = create_cache_context()
    with torch.no_grad(), cache_context(context):
        expected = unet(**kwargs).sample
        if cache_state == "missing":
            del context.buffers["final_output"]
        elif cache_state == "wrong_dtype":
            context.buffers["final_output"] = context.buffers["final_output"].double()
        actual = unet(**kwargs).sample
    hook.remove()
    assert len(later_blocks) == (2 if cache_state == "missing" else 1)
    assert actual is not None
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA to test cache device alignment")
def test_cached_output_moves_to_active_device():
    unet = make_unet().cuda()
    kwargs = unet_inputs()
    kwargs["sample"] = kwargs["sample"].cuda()
    kwargs["encoder_hidden_states"] = kwargs["encoder_hidden_states"].cuda()
    kwargs["added_cond_kwargs"] = {key: value.cuda() for key, value in kwargs["added_cond_kwargs"].items()}
    apply_cache_on_unet(unet, residual_diff_threshold=1.0)
    later_blocks = []
    hook = unet.down_blocks[1].register_forward_pre_hook(lambda *_: later_blocks.append(True))
    context = create_cache_context()
    with torch.no_grad(), cache_context(context):
        expected = unet(**kwargs).sample
        context.buffers["final_output"] = context.buffers["final_output"].cpu()
        actual = unet(**kwargs).sample
    hook.remove()
    assert len(later_blocks) == 1
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected)

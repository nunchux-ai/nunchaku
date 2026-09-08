import pytest

from .utils import run_test


@pytest.mark.parametrize(
    "residual_diff_threshold,use_double_fb_cache,residual_diff_threshold_multi,residual_diff_threshold_single,height,width,num_frames,num_inference_steps,expected_lpips",
    [
        (0.12, False, None, None, 320, 512, 61, 50, 0.3),
    ],
)
def test_hunyuan_video_fbcache(
    residual_diff_threshold: float,
    use_double_fb_cache: bool,
    residual_diff_threshold_multi: float | None,
    residual_diff_threshold_single: float | None,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    expected_lpips: float,
):
    run_test(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        residual_diff_threshold=residual_diff_threshold,
        use_double_fb_cache=use_double_fb_cache,
        residual_diff_threshold_multi=residual_diff_threshold_multi,
        residual_diff_threshold_single=residual_diff_threshold_single,
        expected_lpips=expected_lpips,
    )


@pytest.mark.parametrize(
    "use_double_fb_cache,residual_diff_threshold_multi,residual_diff_threshold_single,height,width,num_frames,num_inference_steps,expected_lpips",
    [
        (True, 0.09, 0.12, 320, 512, 61, 50, 0.3),
    ],
)
def test_hunyuan_video_double_fbcache(
    use_double_fb_cache: bool,
    residual_diff_threshold_multi: float,
    residual_diff_threshold_single: float,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    expected_lpips: float,
):
    run_test(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        use_double_fb_cache=use_double_fb_cache,
        residual_diff_threshold_multi=residual_diff_threshold_multi,
        residual_diff_threshold_single=residual_diff_threshold_single,
        expected_lpips=expected_lpips,
    )

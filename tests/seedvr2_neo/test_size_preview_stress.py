"""Hostile values for the output size preview — `seedvr2_neo/size_preview.py`.

The preview runs as a Gradio event: its values come from Forge's sliders, from the browser (`javascript/seedvr2.js`
reads the img2img image size) or from anyone calling the Gradio endpoint, so they may be None, strings, floats, NaN,
negative, huge or bools. Whatever arrives, the preview must return one line (a raise in the event shows an error toast
and leaves a stale size on screen), and for usable values it must name exactly the size the run gives.
"""
import math
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from seedvr2_neo import size_preview as sp
from seedvr2_neo.upscale_settings import MAX_OUTPUT_LONG_EDGE, SizeMode, parse_upscale_settings

GARBAGE = [None, "", " ", "abc", "1e999", "nan", float("nan"), float("inf"), float("-inf"), -1, 0, -0.5, 0.4, 1e308,
           -1e308, 10 ** 30, True, False, [], [512], {}, (), b"512", object(), complex(1, 1), "💥" * 5_000]
MODES = [mode.value for mode in SizeMode]


def short_id(value):
    """A test id that stays short for huge values (Windows caps an environment variable at 32767 characters)."""
    return repr(value)[:40]


def expected_line_part(size, width, height):
    """What the preview must say for a run output of width x height: the size, or that the image is too thin."""
    if width and height:
        return f"**{width}×{height}** from {size[0]}×{size[1]}"
    return "too thin to upscale"


def run_output_size(size, resolution, size_mode, scale_factor, long_edge):
    """The size the run gives an input of `size` with these raw values: the same parse and target the run uses."""
    settings = parse_upscale_settings(resolution, 0.0, 0.0, True, 512, 128, size_mode, scale_factor, long_edge)
    return settings.target_for(size).output_size(size)


# --- describe_output_size: never raises ------------------------------------------------------------------------------

@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_in_any_control_still_gives_one_line(value):
    for args in ([value, "Scale factor", 2, 2048], [1080, value, 2, 2048], [1080, "Scale factor", value, 2048],
                 [1080, "Long edge", 2, value], [value, value, value, value]):
        line = sp.describe_output_size((512, 384), *args)
        assert line.startswith("Output size: **") and "\n" not in line


@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_as_the_source_width_or_height_gives_a_line(value):
    for size in ((value, 512), (512, value), (value, value), value):
        line = sp.describe_output_size(size, 1080, "Scale factor", 2, 2048)
        assert line.startswith("Output size:") and "\n" not in line


@pytest.mark.parametrize("size", [(1e308, 1), (1, 1e308), (10 ** 400, 1), (1, 10 ** 309)], ids=short_id)
def test_huge_source_sizes_do_not_raise(size):
    assert sp.describe_output_size(size, 1080, "Scale factor", 2, 2048).startswith("Output size:")


@pytest.mark.parametrize("size", [(0, 0), (0, 512), (512, 0), (-512, 384), (512, -384), (0.4, 512), ("0", "0"),
                                  (float("nan"), 512), (512, float("inf")), ("512.5", 384), [None, None], (512,),
                                  (512, 384, 3), "512x384", None, 512])
def test_sizes_the_browser_cannot_mean_name_no_size(size):
    assert sp.describe_output_size(size, 1080, "Scale factor", 2, 2048) == sp.UNKNOWN_SOURCE_NOTE


# --- the preview equals the run ---------------------------------------------------------------------------------------

def test_the_preview_names_the_size_the_run_gives_for_random_values():
    rng = random.Random(2024)
    choices = [None, "x", -5, 0, 1e9, float("nan")]
    for _ in range(4000):
        size = (rng.randint(1, 8000), rng.randint(1, 8000))
        if max(size) / min(size) > 1000:
            continue
        values = [rng.choice([rng.uniform(0, 5000), rng.choice(choices)]),
                  rng.choice(MODES + [m.upper() for m in MODES] + [None, "8K"]),
                  rng.choice([rng.uniform(0, 6), rng.choice(choices)]),
                  rng.choice([rng.uniform(0, 5000), rng.choice(choices)])]
        width, height = run_output_size(size, *values)
        assert expected_line_part(size, width, height) in sp.describe_output_size(size, *values), (size, values)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [(1, 4096), (4096, 1), (16000, 1000), (1000, 16000), (1, 1), (20000, 20000),
                                  (3840, 2160), (1024, 4096)])
def test_extreme_sources_are_previewed_like_the_run(mode, size):
    width, height = run_output_size(size, 1080, mode, 4, 3840)
    line = sp.describe_output_size(size, 1080, mode, 4, 3840)
    assert expected_line_part(size, width, height) in line
    if mode != SizeMode.SHORTEST_EDGE.value:
        assert max(width, height) <= MAX_OUTPUT_LONG_EDGE


@pytest.mark.parametrize("mode", ["Scale factor", "Long edge"])
@pytest.mark.parametrize("size", [(1, 4096), (4096, 1), (2, 3000)])
def test_the_preview_never_names_a_zero_side(mode, size):
    line = sp.describe_output_size(size, 1080, mode, 2, 2048)
    assert "**0×" not in line and "×0**" not in line
    if "**" not in line:
        assert "too thin to upscale" in line


@pytest.mark.parametrize("size", [(16000, 1000), (100, 1000), (384, 384), (100, 300)])
def test_a_named_raise_names_the_shortest_edge_the_run_gives(size):
    line = sp.describe_output_size(size, 1080, "Scale factor", 1, 2048)
    target = parse_upscale_settings(1080, 0, 0, True, 512, 128, "Scale factor", 1, 2048).target_for(size)
    if target.raised_to_minimum:
        assert f"raised to {target.resolution} px" in line
        assert abs(min(target.output_size(size)) - target.resolution) <= 2
    else:
        assert "raised" not in line


def test_a_raise_is_named_only_when_the_run_raises():
    for size in [(384, 384), (100, 300), (512, 512), (1920, 1080), (256, 1024)]:
        for mode in MODES:
            settings = parse_upscale_settings(1080, 0, 0, True, 512, 128, mode, 1.0, 1024)
            named = "raised to" in sp.describe_output_size(size, 1080, mode, 1.0, 1024)
            assert named == settings.target_for(size).raised_to_minimum, (size, mode)


def test_concurrent_previews_with_different_modes_do_not_mix():
    """Gradio runs events of several browser tabs in threads; the preview must hold no state between calls."""
    cases = [((rng_w, rng_h), mode, factor, edge) for rng_w, rng_h in [(512, 384), (1920, 1080), (300, 900)]
             for mode in MODES for factor in (1.0, 2.5, 4.0) for edge in (1024, 3840)]
    expected = [sp.describe_output_size(size, 1080, mode, factor, edge) for size, mode, factor, edge in cases]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(20):
            got = list(pool.map(lambda case: sp.describe_output_size(case[0], 1080, *case[1:]), cases))
            assert got == expected


# --- txt2img_final_size --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_in_any_txt2img_control_gives_a_size_or_none(value):
    base = dict(width=512, height=384, enable_hr=True, hr_scale=2.0, hr_resize_x=0, hr_resize_y=0, res_step=64)
    for name in base:
        result = sp.txt2img_final_size(**{**base, name: value})
        assert result is None or (isinstance(result, tuple) and len(result) == 2 and all(
            isinstance(side, int) and side > 0 for side in result)), (name, value, result)


@pytest.mark.parametrize("hr_scale", [0, -1, -2.5, 0.01, float("nan"), float("inf"), 1e308, "abc", None])
def test_hires_scales_that_give_no_image_give_none(hr_scale):
    assert sp.txt2img_final_size(512, 384, True, hr_scale, 0, 0) is None


@pytest.mark.parametrize("resize_x, resize_y", [(-64, 0), (0, -64), (-64, -64), (16, 0), (0, 16)])
def test_negative_or_tiny_hires_targets_give_none(resize_x, resize_y):
    assert sp.txt2img_final_size(512, 384, True, 2.0, resize_x, resize_y) is None


@pytest.mark.parametrize("res_step", [0, -64, "abc", None, float("nan"), 10 ** 30])
def test_unusable_res_step_never_raises(res_step):
    result = sp.txt2img_final_size(512, 384, True, 1.5, 0, 0, res_step)
    assert result is None or all(side > 0 for side in result)


def test_huge_hires_targets_still_preview_without_raising():
    source = sp.txt2img_final_size(1, 2048, True, 1.0, 10 ** 300, 0)
    line = sp.describe_output_size(source, 1080, "Scale factor", 4, 3840)
    assert line.startswith("Output size:")


def test_a_txt2img_preview_equals_the_run_for_every_hires_combination():
    for width, height in [(512, 384), (832, 1216), (1024, 1024), (256, 2048)]:
        for enable_hr, hr_scale, resize_x, resize_y in [(False, 2, 0, 0), (True, 1.5, 0, 0), (True, 2, 1000, 0),
                                                        (True, 2, 0, 1000), (True, 2, 1000, 700)]:
            source = sp.txt2img_final_size(width, height, enable_hr, hr_scale, resize_x, resize_y)
            assert source is not None and all(side % 64 == 0 for side in source)
            for mode in MODES:
                out_w, out_h = run_output_size(source, 1080, mode, 2, 2048)
                assert f"**{out_w}×{out_h}**" in sp.describe_output_size(source, 1080, mode, 2, 2048)

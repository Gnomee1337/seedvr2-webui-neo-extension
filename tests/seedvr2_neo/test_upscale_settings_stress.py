"""Hostile values for the output size modes — `seedvr2_neo/upscale_settings.py`.

`size_mode`, `scale_factor` and `long_edge` come from the UI or from any API caller's `script_args`, and the input size
from any image an API caller uploads. Whatever arrives, the settings stay inside the sliders' ranges, and the output
size the run gets for an input is a real image size: both sides positive and even, the long edge at most
`MAX_OUTPUT_LONG_EDGE` in the scale factor and long edge modes.

A shortest edge under 512 px is raised towards 512 only while the long edge stays within the mode's cap, so a very wide
panorama keeps the cap and a shortest edge under 512. Past an aspect ratio of about cap / 2 (1920:1 at 3840) the short
side rounds to 0 px: such an image cannot be upscaled, the run returns it with `SeedVR2 Error` and the preview says so.

`OutputTarget.output_size` is also checked against the vendored `SideResize` the pipeline really resizes with, so the
size the run plans (pass budget, console line, preview) is the size it gets.
"""
import importlib
import itertools
import math
import random
import sys
import types
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
import torch

from seedvr2_neo.upscale_settings import FALLBACK_SIZE_MODE, LONG_EDGE, MAX_LONG_EDGE_FACTOR, MAX_OUTPUT_LONG_EDGE, \
    RESOLUTION, SCALE_FACTOR, OutputTarget, SizeMode, UpscaleSettings, parse_size_mode, parse_upscale_settings

GARBAGE = [None, "", "   ", "abc", "💥" * 10_000, "1e999", "-inf", "nan", float("nan"), float("inf"), float("-inf"),
           -1, 0, -0.0, 10 ** 400, -10 ** 400, 1e308, True, False, [], [2.0], {}, (), b"2", object(), complex(2, 1),
           np.float32("nan"), np.int64(-5)]

# Inputs an API caller can upload: extreme panoramas, one-pixel lines, odd sizes, sizes past the output cap.
SIZES = [(1, 1), (1, 2), (2, 1), (7, 3), (64, 64), (384, 384), (511, 513), (512, 384), (1000, 16000), (16000, 1000),
         (1024, 4096), (4096, 1024), (4097, 1024), (1920, 1080), (3840, 2160), (2160, 3840), (5000, 4999),
         (8192, 8192), (20000, 20000), (12, 4096), (4096, 12), (1, 1300), (1300, 1)]

# Aspect ratios so extreme that the short output side rounds to 0 px in every mode at the default settings.
ZERO_EDGE_SIZES = [(1, 4096), (4096, 1), (1, 20000), (3, 20000)]

MODES = list(SizeMode)


def short_id(value):
    """A test id that stays short for huge values (Windows caps an environment variable at 32767 characters)."""
    return repr(value)[:40]


def settings(mode, scale_factor=2.0, long_edge=2048, resolution=1080):
    return parse_upscale_settings(resolution, 0.0, 0.0, True, 512, 128, mode, scale_factor, long_edge)


def cap_of(parsed):
    """The longest output edge the mode allows: the pipeline's `max_resolution` for this target."""
    if parsed.size_mode is SizeMode.SCALE_FACTOR:
        return MAX_OUTPUT_LONG_EDGE
    if parsed.size_mode is SizeMode.LONG_EDGE:
        return parsed.long_edge
    return parsed.resolution * MAX_LONG_EDGE_FACTOR


def zero_side_possible(parsed, size):
    """The short output side is under 2 px (0 after the even rounding) only when the long edge cap leaves no room."""
    return cap_of(parsed) * max(1, min(size)) / max(1, max(size)) < 2 + 1e-9


def all_settings():
    for mode in MODES:
        for factor, long_edge, resolution in [(1.0, 1024, 512), (2.0, 2048, 1080), (4.0, 3840, 3840),
                                              (1.37, 1088, 2000)]:
            yield settings(mode, factor, long_edge, resolution)


# --- parse_upscale_settings: every new argument, any value ------------------------------------------------------------

@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_in_any_size_argument_gives_settings_inside_the_ranges(value):
    for kwargs in ({"size_mode": value}, {"scale_factor": value}, {"long_edge": value},
                   {"size_mode": value, "scale_factor": value, "long_edge": value}):
        parsed = parse_upscale_settings(1080, 0, 0, True, 512, 128, **kwargs)
        assert isinstance(parsed.size_mode, SizeMode)
        assert SCALE_FACTOR.minimum <= parsed.scale_factor <= SCALE_FACTOR.maximum
        assert type(parsed.long_edge) is int and LONG_EDGE.minimum <= parsed.long_edge <= LONG_EDGE.maximum
        assert type(parsed.scale_factor) is float and math.isfinite(parsed.scale_factor)


@pytest.mark.parametrize("value", [None, "abc", float("nan"), float("inf"), True, [], object()], ids=short_id)
def test_unreadable_values_take_the_slider_defaults(value):
    parsed = parse_upscale_settings(1080, 0, 0, True, 512, 128, "Scale factor", value, value)
    assert (parsed.scale_factor, parsed.long_edge) == (SCALE_FACTOR.default, LONG_EDGE.default)


@pytest.mark.parametrize("value, expected", [(-1, 1.0), (0, 1.0), (0.999, 1.0), (4.0001, 4.0), (1e308, 4.0),
                                             ("3.5", 3.5), (" 2 ", 2.0), (Decimal("2.345"), 2.35),
                                             (Fraction(7, 2), 3.5), (np.float32(1.5), 1.5), (3.999, 4.0),
                                             (1.004999, 1.0)])
def test_scale_factor_is_clamped_and_rounded_to_two_decimals(value, expected):
    assert settings("Scale factor", scale_factor=value).scale_factor == pytest.approx(expected)


@pytest.mark.parametrize("value, expected", [(-1, 1024), (0, 1024), (1023.9, 1024), (3840.5, 3840), (10 ** 9, 3840),
                                             ("2048", 2048), ("2048.9", 2048), (np.int64(1500), 1500)])
def test_long_edge_is_clamped_to_its_slider(value, expected):
    assert settings("Long edge", long_edge=value).long_edge == expected


# --- parse_size_mode -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("Scale factor", SizeMode.SCALE_FACTOR), ("  scale FACTOR\t", SizeMode.SCALE_FACTOR),
    ("LONG EDGE", SizeMode.LONG_EDGE), ("\nshortest edge\n", SizeMode.SHORTEST_EDGE),
    (SizeMode.LONG_EDGE, SizeMode.LONG_EDGE),
    # look-alikes and near misses: a caller must not get a mode it did not name
    ("Long edge", FALLBACK_SIZE_MODE), ("Long еdge", FALLBACK_SIZE_MODE), ("Long_edge", FALLBACK_SIZE_MODE),
    ("Longedge", FALLBACK_SIZE_MODE), ("LONG_EDGE", FALLBACK_SIZE_MODE), ("SizeMode.LONG_EDGE", FALLBACK_SIZE_MODE),
    ("Scale factor, Steps: 999", FALLBACK_SIZE_MODE), ("Scale factor\nSteps: 999", FALLBACK_SIZE_MODE),
    (b"Long edge", FALLBACK_SIZE_MODE), (["Long edge"], FALLBACK_SIZE_MODE), (0, FALLBACK_SIZE_MODE),
    (2, FALLBACK_SIZE_MODE), (None, FALLBACK_SIZE_MODE), ("", FALLBACK_SIZE_MODE), ("x" * 100_000, FALLBACK_SIZE_MODE),
], ids=short_id)
def test_size_mode_is_named_exactly_or_falls_back(value, expected):
    assert parse_size_mode(value) is expected


def test_a_str_subclass_naming_a_mode_is_read_like_a_string():
    class Label(str):
        pass

    assert parse_size_mode(Label("Long edge")) is SizeMode.LONG_EDGE


# --- target_for / output_size: extreme input sizes in every mode --------------------------------------------------

@pytest.mark.parametrize("size", SIZES)
def test_every_mode_gives_a_real_even_size_inside_the_caps(size):
    for parsed in all_settings():
        target = parsed.target_for(size)
        width, height = target.output_size(size)
        assert (width > 0 and height > 0) or zero_side_possible(parsed, size), (parsed.size_mode, size, (width, height))
        assert width % 2 == 0 and height % 2 == 0
        assert target.max_resolution == cap_of(parsed)
        assert max(width, height) <= target.max_resolution
        assert 1 <= target.resolution <= RESOLUTION.maximum
        if parsed.size_mode is SizeMode.SHORTEST_EDGE:
            assert target.resolution >= RESOLUTION.minimum
        else:
            assert max(width, height) <= MAX_OUTPUT_LONG_EDGE
        assert target.output_pixels(size) == width * height


@pytest.mark.parametrize("size", SIZES)
def test_a_shortest_edge_under_512_only_where_the_long_edge_cap_leaves_no_room(size):
    for parsed in all_settings():
        if parsed.size_mode is SizeMode.SHORTEST_EDGE:
            continue
        target = parsed.target_for(size)
        if target.resolution < RESOLUTION.minimum:
            # raising further would push the long edge past the cap
            assert target.resolution * max(size) / min(size) >= cap_of(parsed) - max(size) / min(size) - 2, \
                (parsed.size_mode, size, target)


@pytest.mark.parametrize("mode, size", [(mode, size) for mode in MODES for size in ZERO_EDGE_SIZES])
def test_a_zero_side_happens_only_when_the_cap_leaves_under_two_pixels(mode, size):
    parsed = settings(mode)
    width, height = parsed.target_for(size).output_size(size)
    if width == 0 or height == 0:
        assert zero_side_possible(parsed, size)


@pytest.mark.parametrize("size", SIZES + ZERO_EDGE_SIZES)
def test_output_orientation_follows_the_input(size):
    for parsed in all_settings():
        width, height = parsed.target_for(size).output_size(size)
        if size[0] < size[1]:
            assert width <= height
        elif size[0] > size[1]:
            assert width >= height
        else:
            assert width == height


@pytest.mark.parametrize("size", [(0, 0), (-5, 100), (100, -5), (0, 512), (10 ** 6, 0)])
def test_zero_or_negative_sides_are_read_as_one_pixel(size):
    for parsed in all_settings():
        target = parsed.target_for(size)
        assert 1 <= target.resolution <= RESOLUTION.maximum
        clean = tuple(max(1, side) for side in size)
        assert target == parsed.target_for(clean)
        assert target.output_size(size) == target.output_size(clean)


@pytest.mark.parametrize("size, scale_factor", [((384, 384), 1.0), ((100, 300), 1.0), ((200, 100), 2.0),
                                                ((1, 1), 4.0)])
def test_raised_flag_is_set_when_the_shortest_edge_is_raised_to_512(size, scale_factor):
    target = settings("Scale factor", scale_factor=scale_factor).target_for(size)
    assert target.raised_to_minimum
    assert min(target.output_size(size)) == RESOLUTION.minimum


@pytest.mark.parametrize("mode, size", [("Scale factor", (16000, 1000)), ("Scale factor", (8000, 1000)),
                                        ("Long edge", (1000, 16000)), ("Long edge", (5000, 600)),
                                        ("Scale factor", (100, 1000))])
def test_a_raise_stopped_by_the_long_edge_cap_names_the_real_shortest_edge(mode, size):
    """16000x1000 at x1 cannot reach a 512 px shortest edge within 3840 px: it keeps the cap, and a raise, when there
    is one, names the shortest edge it really reached."""
    target = settings(mode, scale_factor=1.0, long_edge=2048).target_for(size)
    out = target.output_size(size)
    assert max(out) <= target.max_resolution
    if target.raised_to_minimum:
        assert f"raised to {target.resolution} px" in target.raise_note()
        assert abs(min(out) - target.resolution) <= 2
    else:
        assert target.raise_note() == ""


def test_the_shortest_edge_mode_never_reports_a_raise():
    for size in SIZES + ZERO_EDGE_SIZES:
        assert not settings("Shortest edge", resolution=512).target_for(size).raised_to_minimum


def test_scale_factor_follows_the_input_within_rounding_for_ordinary_images():
    rng = random.Random(11)
    for _ in range(3000):
        short = rng.randint(256, 2048)
        size = (short, int(short * rng.uniform(1.0, MAX_LONG_EDGE_FACTOR)))
        if rng.random() < 0.5:
            size = size[::-1]
        factor = round(rng.uniform(1.0, 4.0), 2)
        target = settings("Scale factor", scale_factor=factor).target_for(size)
        if target.raised_to_minimum:
            continue
        expected_long = min(factor * max(size), MAX_OUTPUT_LONG_EDGE)
        ratio = max(size) / min(size)
        assert max(target.output_size(size)) <= MAX_OUTPUT_LONG_EDGE
        assert abs(max(target.output_size(size)) - expected_long) <= ratio / 2 + 2, (size, factor)


def test_a_larger_scale_factor_never_gives_a_smaller_image():
    for size in [(512, 384), (100, 1000), (1920, 1080), (3000, 2999), (777, 333)]:
        sizes = [settings("Scale factor", scale_factor=tenths / 10).target_for(size).output_size(size)
                 for tenths in range(10, 41)]
        assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(sizes, sizes[1:])), size


def test_long_edge_mode_never_exceeds_the_chosen_long_edge():
    """The chosen long edge is the pipeline's cap, so even a raise towards 512 px stops at it."""
    rng = random.Random(5)
    for _ in range(3000):
        size = (rng.randint(1, 9000), rng.randint(1, 9000))
        long_edge = rng.randrange(1024, 3841, 64)
        target = settings("Long edge", long_edge=long_edge).target_for(size)
        assert max(target.output_size(size)) <= long_edge, (size, long_edge)


def test_a_larger_long_edge_never_gives_a_smaller_image():
    for size in [(512, 384), (100, 1000), (1920, 1080), (3000, 2999), (777, 333)]:
        sizes = [settings("Long edge", long_edge=edge).target_for(size).output_size(size)
                 for edge in range(1024, 3841, 64)]
        assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(sizes, sizes[1:])), size


def test_random_inputs_and_settings_never_break_the_caps():
    rng = random.Random(1234)
    for _ in range(5000):
        size = (rng.choice([1, 2, 3, rng.randint(1, 20000)]), rng.randint(1, 20000))
        parsed = settings(rng.choice(MODES + ["garbage", None]), rng.uniform(-10, 10), rng.uniform(0, 10_000),
                          rng.uniform(0, 10_000))
        target = parsed.target_for(size)
        width, height = target.output_size(size)
        assert width % 2 == 0 and height % 2 == 0
        assert (width > 0 and height > 0) or zero_side_possible(parsed, size), (parsed.size_mode, size)
        if parsed.size_mode is not SizeMode.SHORTEST_EDGE:
            assert max(width, height) <= MAX_OUTPUT_LONG_EDGE


# --- state between runs --------------------------------------------------------------------------------------------

def test_settings_and_targets_hold_no_shared_state():
    first = settings("Scale factor", scale_factor=4.0)
    second = settings("Long edge", long_edge=1024)
    before = first.target_for((512, 384))
    second.target_for((512, 384))
    assert first.target_for((512, 384)) == before
    with pytest.raises(Exception):
        first.size_mode = SizeMode.LONG_EDGE  # frozen: one run cannot change another's settings


def test_the_defaults_of_the_dataclass_keep_old_callers_on_the_shortest_edge():
    old = UpscaleSettings(resolution=1536, input_noise=0, latent_noise=0, use_tile_vae=True, tile_size=512,
                          tile_overlap=128)
    assert old.size_mode is SizeMode.SHORTEST_EDGE and old.target_for((10, 20)).resolution == 1536


# --- output_size against the vendored resize --------------------------------------------------------------------------

@pytest.fixture
def side_resize(monkeypatch):
    """The vendored `SideResize`, with its only heavy import (`memory_manager`, needs psutil) stubbed; the modules it
    loads are dropped afterwards so other tests see a clean `sys.modules`."""
    before = set(sys.modules)
    stub = types.ModuleType("seedvr2_vendor.optimization.memory_manager")
    stub.is_mps_available = lambda: False
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    try:
        module = importlib.import_module("seedvr2_vendor.data.image.transforms.side_resize")
    except Exception as error:  # pragma: no cover - depends on the environment
        pytest.skip(f"vendored SideResize not importable here: {error!r}")
    yield module.SideResize
    for name in set(sys.modules) - before:
        if name.startswith("seedvr2_vendor") and name != stub.__name__:
            sys.modules.pop(name, None)


def vendor_output_size(side_resize, target: OutputTarget, size):
    """The (width, height) the pipeline really gives: SideResize, then the even rounding of `true_target_dims`."""
    from torchvision.transforms import InterpolationMode

    width, height = size
    frame = torch.zeros(1, height, width, dtype=torch.uint8)
    resized = side_resize(target.resolution, target.max_resolution, interpolation=InterpolationMode.NEAREST)(frame)
    out_height, out_width = resized.shape[-2:]
    return out_width // 2 * 2, out_height // 2 * 2


@pytest.mark.parametrize("size", [(512, 384), (384, 512), (511, 513), (1920, 1080), (1080, 1920), (777, 333),
                                  (1024, 4096), (4096, 1024), (4097, 1024), (3000, 600), (100, 2000), (64, 64),
                                  (3841, 2161), (13, 7), (1, 1)])
@pytest.mark.parametrize("mode", MODES)
def test_planned_output_size_is_what_the_vendored_resize_gives(side_resize, mode, size):
    for factor, long_edge, resolution in [(1.0, 1024, 512), (2.0, 2048, 1080), (4.0, 3840, 1536), (1.37, 1088, 720)]:
        target = settings(mode, factor, long_edge, resolution).target_for(size)
        assert target.output_size(size) == vendor_output_size(side_resize, target, size), (mode, size, target)

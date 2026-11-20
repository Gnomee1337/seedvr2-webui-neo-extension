"""Hostile values for the output size modes — `seedvr2_neo/upscale_settings.py`.

`size_mode`, `scale_factor` and `long_edge` come from the UI or from any API caller's `script_args`, and the input size
from any image an API caller uploads. Whatever arrives, the settings stay inside the sliders' ranges, and the output
size the run gets for an input is a real image size: both sides positive and even, the long edge at most
`MAX_OUTPUT_LONG_EDGE` in the scale factor and long edge modes.

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

# Aspect ratios past ~1365:1 (the 512 px shortest edge scaled down to fit a 2048 px long edge rounds to 0 or 1 px).
ZERO_EDGE_SIZES = [(1, 4096), (4096, 1), (2, 3000), (3000, 2), (1, 1400), (16000, 10)]
# The shortest edge mode caps the long edge at 4 x 1080 = 4320 px instead, so it gets a 0 px side only past ~2900:1.
ZERO_EDGE_CASES = [(mode, size) for mode in (SizeMode.SCALE_FACTOR, SizeMode.LONG_EDGE) for size in ZERO_EDGE_SIZES] +     [(SizeMode.SHORTEST_EDGE, size) for size in [(1, 4096), (4096, 1)]]

MODES = list(SizeMode)


def short_id(value):
    """A test id that stays short for huge values (Windows caps an environment variable at 32767 characters)."""
    return repr(value)[:40]


def settings(mode, scale_factor=2.0, long_edge=2048, resolution=1080):
    return parse_upscale_settings(resolution, 0.0, 0.0, True, 512, 128, mode, scale_factor, long_edge)


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
        assert width > 0 and height > 0, (parsed.size_mode, size, (width, height))
        assert width % 2 == 0 and height % 2 == 0
        assert RESOLUTION.minimum <= target.resolution <= RESOLUTION.maximum
        assert max(width, height) <= target.max_resolution
        if parsed.size_mode is SizeMode.SHORTEST_EDGE:
            assert target.max_resolution == target.resolution * MAX_LONG_EDGE_FACTOR
        else:
            assert max(width, height) <= MAX_OUTPUT_LONG_EDGE
        assert target.output_pixels(size) == width * height


@pytest.mark.xfail(strict=True, reason="BUG: past ~1365:1 the planned output has a 0 px side (1x4096 -> 0x2048); the "
                                       "vendored SideResize then raises, so the run ends with 'SeedVR2 Error'")
@pytest.mark.parametrize("mode, size", ZERO_EDGE_CASES)
def test_extreme_aspect_ratios_never_give_a_zero_side(mode, size):
    width, height = settings(mode).target_for(size).output_size(size)
    assert width > 0 and height > 0


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
        assert RESOLUTION.minimum <= target.resolution <= RESOLUTION.maximum
        clean = tuple(max(1, side) for side in size)
        assert target.output_size(size) == target.output_size(clean)


@pytest.mark.parametrize("size, scale_factor", [((384, 384), 1.0), ((100, 300), 1.0), ((200, 100), 2.0),
                                                ((1, 1), 4.0)])
def test_raised_flag_is_set_only_when_the_shortest_edge_is_raised(size, scale_factor):
    target = settings("Scale factor", scale_factor=scale_factor).target_for(size)
    assert target.raised_to_minimum
    assert min(target.output_size(size)) == RESOLUTION.minimum


@pytest.mark.xfail(strict=True, reason="BUG: a panorama past 4:1 whose shortest edge is 'raised to 512' is then scaled "
                                       "down by the 4x long edge cap: 16000x1000 at x2 -> 2048x128, while the console, "
                                       "preview and infotext say 512 / 'raised to the 512 px minimum'")
@pytest.mark.parametrize("mode, size", [("Scale factor", (16000, 1000)), ("Scale factor", (8000, 1000)),
                                        ("Long edge", (1000, 16000)), ("Long edge", (5000, 600))])
def test_a_raised_shortest_edge_is_really_at_least_512(mode, size):
    target = settings(mode, scale_factor=1.0, long_edge=2048).target_for(size)
    assert target.raised_to_minimum
    assert min(target.output_size(size)) >= RESOLUTION.minimum


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


def test_long_edge_mode_never_exceeds_the_chosen_long_edge_unless_the_shortest_edge_is_raised():
    """A raise to 512 px may pass the chosen long edge (1920x800 at 1024 -> 1228x512): that is the documented minimum,
    and the console and preview name the raise. The long edge stays within the 4 x 512 cap then."""
    rng = random.Random(5)
    for _ in range(3000):
        size = (rng.randint(1, 9000), rng.randint(1, 9000))
        long_edge = rng.randrange(1024, 3841, 64)
        target = settings("Long edge", long_edge=long_edge).target_for(size)
        limit = RESOLUTION.minimum * MAX_LONG_EDGE_FACTOR if target.raised_to_minimum else long_edge
        assert max(target.output_size(size)) <= max(limit, long_edge), (size, long_edge)


def test_a_larger_long_edge_never_gives_a_smaller_image():
    for size in [(512, 384), (100, 1000), (1920, 1080), (3000, 2999), (777, 333)]:
        sizes = [settings("Long edge", long_edge=edge).target_for(size).output_size(size)
                 for edge in range(1024, 3841, 64)]
        assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(sizes, sizes[1:])), size


def test_random_inputs_and_settings_never_break_the_caps():
    rng = random.Random(1234)
    for _ in range(5000):
        size = (rng.choice([1, 2, 3, rng.randint(1, 20000)]), rng.randint(1, 20000))
        if max(size) / min(size) > 1000:  # the zero side past ~1365:1 is the xfail above
            continue
        parsed = settings(rng.choice(MODES + ["garbage", None]), rng.uniform(-10, 10), rng.uniform(0, 10_000),
                          rng.uniform(0, 10_000))
        target = parsed.target_for(size)
        width, height = target.output_size(size)
        assert width > 0 and height > 0 and width % 2 == 0 and height % 2 == 0
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

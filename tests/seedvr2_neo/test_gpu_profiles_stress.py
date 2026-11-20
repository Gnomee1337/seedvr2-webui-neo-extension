"""Hostile labels and the consistency of the GPU profiles — `seedvr2_neo/gpu_profiles.py`.

The label comes from a Gradio dropdown, which a Gradio API client can call with any value. Checked here: only an exact
label picks a profile, nothing raises, and every profile's values are ones the sliders and the run accept unchanged
(inside the bounds, on the slider step, overlap at most half the tile, BlockSwap inside its range).
"""
import math

import pytest

from seedvr2_neo.gpu_profiles import GPU_PROFILES, MANUAL, describe_profiles, find_profile, profile_choices
from seedvr2_neo.memory_plan import MAX_BLOCKS_TO_SWAP, build_memory_plan
from seedvr2_neo.upscale_settings import TILE_OVERLAP, TILE_SIZE, parse_upscale_settings

JUNK_LABELS = [None, "", " ", MANUAL, "manual", "8gb", "8 gb", " 8 GB", "8 GB ", "8 GB\n", "8 GB", "８ GB", "8 GB\x00",
               b"8 GB", bytearray(b"8 GB"), ["8 GB"], ("8 GB",), {"8 GB": 1}, {"label": "8 GB"}, 8, 8.0, float("nan"),
               True, object(), "💥" * 10_000, "8 GB" * 100_000, "16 GB", "6 GB", "Auto"]


def short_id(value):
    return repr(value)[:30]


@pytest.mark.parametrize("label", JUNK_LABELS, ids=short_id)
def test_anything_but_an_exact_label_is_manual(label):
    assert find_profile(label) is None


@pytest.mark.parametrize("profile", GPU_PROFILES, ids=lambda profile: profile.label)
def test_an_exact_label_finds_its_profile(profile):
    assert find_profile(profile.label) is profile
    assert find_profile(str(profile.label)) is profile


def test_a_str_subclass_label_finds_the_profile():
    class Label(str):
        pass

    assert find_profile(Label("12 GB")).label == "12 GB"


def test_choices_are_unique_manual_first_and_every_label_resolves():
    choices = profile_choices()
    assert choices[0] == MANUAL and len(choices) == len(set(choices))
    assert [find_profile(choice) for choice in choices[1:]] == list(GPU_PROFILES)
    assert profile_choices() is not profile_choices()  # a fresh list: Gradio may keep and mutate it


def on_step(value, bounds):
    steps = (value - bounds.minimum) / bounds.step
    return math.isclose(steps, round(steps))


@pytest.mark.parametrize("profile", GPU_PROFILES, ids=lambda profile: profile.label)
def test_every_profile_value_fits_the_sliders_and_survives_parsing(profile):
    assert TILE_SIZE.minimum <= profile.tile_size <= TILE_SIZE.maximum and on_step(profile.tile_size, TILE_SIZE)
    assert TILE_OVERLAP.minimum <= profile.tile_overlap <= TILE_OVERLAP.maximum
    assert on_step(profile.tile_overlap, TILE_OVERLAP)
    assert profile.tile_overlap <= profile.tile_size // 2
    for is_7b in (False, True):
        blocks = profile.blocks_to_swap(is_7b)
        assert isinstance(blocks, int) and 0 <= blocks <= MAX_BLOCKS_TO_SWAP
        assert build_memory_plan(None, blocks, False).block_swap_config()["blocks_to_swap"] == blocks
    settings = parse_upscale_settings(1080, 0.0, 0.0, True, profile.tile_size, profile.tile_overlap)
    assert (settings.tile_size, settings.tile_overlap) == (profile.tile_size, profile.tile_overlap)


def test_bigger_gpus_never_get_smaller_tiles_or_more_swapped_blocks():
    for smaller, bigger in zip(GPU_PROFILES, GPU_PROFILES[1:]):
        assert bigger.tile_size >= smaller.tile_size
        assert bigger.blocks_to_swap(True) <= smaller.blocks_to_swap(True)
        assert bigger.blocks_to_swap(False) <= smaller.blocks_to_swap(False)
    for profile in GPU_PROFILES:
        assert profile.blocks_to_swap(True) >= profile.blocks_to_swap(False)


@pytest.mark.parametrize("is_7b", [0, 1, "", "yes", None, []])
def test_truthy_or_falsy_size_flags_pick_the_3b_or_7b_count(is_7b):
    for profile in GPU_PROFILES:
        expected = profile.blocks_to_swap_7b if is_7b else profile.blocks_to_swap_3b
        assert profile.blocks_to_swap(is_7b) == expected


def test_the_hint_names_every_unmeasured_profile_and_only_those():
    hint = describe_profiles()
    for profile in GPU_PROFILES:
        assert (profile.label in hint) == (not profile.measured)


def test_profiles_are_frozen():
    with pytest.raises(Exception):
        GPU_PROFILES[0].tile_size = 4096

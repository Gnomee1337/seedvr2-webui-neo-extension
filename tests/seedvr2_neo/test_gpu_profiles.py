"""GPU profiles that fill the memory sliders — `seedvr2_neo/gpu_profiles.py`."""

import pytest

from seedvr2_neo.gpu_profiles import GPU_PROFILES, MANUAL, describe_profiles, find_profile, profile_choices
from seedvr2_neo.memory_plan import MAX_BLOCKS_TO_SWAP, RECOMMENDED_7B_BLOCKS_TO_SWAP
from seedvr2_neo.upscale_settings import TILE_OVERLAP, TILE_SIZE


def test_manual_comes_first_then_the_profiles_from_the_smallest_gpu():
    assert profile_choices() == [MANUAL, "8 GB", "12 GB", "24 GB"]


def test_the_8_gb_profile_is_the_measured_defaults():
    profile = find_profile("8 GB")
    assert profile.measured
    assert (profile.tile_size, profile.tile_overlap) == (TILE_SIZE.default, TILE_OVERLAP.default)
    assert (profile.blocks_to_swap(False), profile.blocks_to_swap(True)) == (0, RECOMMENDED_7B_BLOCKS_TO_SWAP)


@pytest.mark.parametrize("profile", GPU_PROFILES, ids=lambda profile: profile.label)
def test_every_profile_value_is_one_the_sliders_can_show(profile):
    assert TILE_SIZE.minimum <= profile.tile_size <= TILE_SIZE.maximum
    assert (profile.tile_size - TILE_SIZE.minimum) % TILE_SIZE.step == 0
    assert TILE_OVERLAP.minimum <= profile.tile_overlap <= profile.tile_size // 2
    for is_7b in (False, True):
        assert 0 <= profile.blocks_to_swap(is_7b) <= MAX_BLOCKS_TO_SWAP


def test_bigger_gpus_never_get_smaller_tiles_or_more_blockswap():
    for smaller, bigger in zip(GPU_PROFILES, GPU_PROFILES[1:]):
        assert bigger.tile_size >= smaller.tile_size
        assert bigger.blocks_to_swap(True) <= smaller.blocks_to_swap(True)


@pytest.mark.parametrize("label", [MANUAL, None, "", "8gb", "16 GB", 8, ["8 GB"]])
def test_manual_and_unknown_labels_are_no_profile(label):
    assert find_profile(label) is None


def test_the_hint_names_the_unmeasured_profiles():
    hint = describe_profiles()
    assert "12 GB and 24 GB values are not measured" in hint and "8 GB" not in hint

"""Color correction and attention backend from raw UI / API values — `seedvr2_neo/pipeline_options.py`."""

import pytest

from seedvr2_neo.pipeline_options import ATTENTION_PACKAGES, DEFAULT_ATTENTION_MODE, DEFAULT_COLOR_CORRECTION, \
    AttentionMode, ColorCorrection, PipelineOptions, choice_values, installed_attention_modes, parse_attention_mode, \
    parse_color_correction, resolve_attention_mode


def test_the_defaults_are_what_every_run_used_before_the_choice():
    assert PipelineOptions() == PipelineOptions(ColorCorrection.LAB, AttentionMode.SDPA)
    assert (DEFAULT_COLOR_CORRECTION.value, DEFAULT_ATTENTION_MODE.value) == ("lab", "sdpa")


def test_the_choices_are_the_vendored_pipeline_strings():
    assert choice_values(ColorCorrection) == ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]
    assert choice_values(AttentionMode) == ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]


@pytest.mark.parametrize("value, expected", [
    ("wavelet", ColorCorrection.WAVELET), (" HSV ", ColorCorrection.HSV), ("None", ColorCorrection.NONE),
    (ColorCorrection.ADAIN, ColorCorrection.ADAIN), ("Wavelet_Adaptive", ColorCorrection.WAVELET_ADAPTIVE),
])
def test_a_color_correction_is_read_by_name(value, expected):
    assert parse_color_correction(value) is expected


@pytest.mark.parametrize("value", [None, "", "rgb", 3, 1.5, True, ["lab"], {"lab": 1}, "lab\x00", "l" * 10000])
def test_an_unreadable_color_correction_is_lab(value):
    assert parse_color_correction(value) is ColorCorrection.LAB


@pytest.mark.parametrize("value, expected", [
    ("sageattn_2", AttentionMode.SAGEATTN_2), ("FLASH_ATTN_3", AttentionMode.FLASH_ATTN_3),
    (AttentionMode.SDPA, AttentionMode.SDPA),
])
def test_an_attention_mode_is_read_by_name(value, expected):
    assert parse_attention_mode(value) is expected


@pytest.mark.parametrize("value", [None, "", "flash", "xformers", 0, False, ColorCorrection.LAB])
def test_an_unreadable_attention_mode_is_sdpa(value):
    assert parse_attention_mode(value) is AttentionMode.SDPA


def test_only_installed_backends_are_listed_after_sdpa():
    installed = {"sageattention", "flash_attn"}
    modes = installed_attention_modes(lambda name: object() if name in installed else None)
    assert modes == [AttentionMode.SDPA, AttentionMode.FLASH_ATTN_2, AttentionMode.SAGEATTN_2]


def test_with_no_packages_only_sdpa_is_listed():
    assert installed_attention_modes(lambda name: None) == [AttentionMode.SDPA]


@pytest.mark.parametrize("error", [ImportError("broken parent"), ValueError("__spec__ is None")])
def test_a_package_whose_lookup_fails_is_not_listed(error):
    def find_spec(name):
        if name == "flash_attn":
            raise error
        return object() if name == "sageattn3" else None

    assert installed_attention_modes(find_spec) == [AttentionMode.SDPA, AttentionMode.SAGEATTN_3]


def test_every_backend_but_sdpa_names_its_package():
    assert set(ATTENTION_PACKAGES) == set(AttentionMode) - {AttentionMode.SDPA}


def test_the_real_lookup_runs_without_importing_anything():
    assert installed_attention_modes()[0] is AttentionMode.SDPA


def test_an_available_backend_is_used_as_requested():
    assert resolve_attention_mode(AttentionMode.SAGEATTN_2, {AttentionMode.SAGEATTN_2: True}) == \
        (AttentionMode.SAGEATTN_2, None)


def test_sdpa_needs_no_availability_entry():
    assert resolve_attention_mode(AttentionMode.SDPA, {}) == (AttentionMode.SDPA, None)


@pytest.mark.parametrize("available", [{}, {AttentionMode.FLASH_ATTN_2: False}, {AttentionMode.SAGEATTN_2: True}])
def test_an_unavailable_backend_falls_back_to_sdpa_with_a_note(available):
    mode, note = resolve_attention_mode(AttentionMode.FLASH_ATTN_2, available)
    assert mode is AttentionMode.SDPA
    assert "flash_attn_2" in note and "sdpa" in note

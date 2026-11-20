"""Hostile values for color correction and the attention backend — `seedvr2_neo/pipeline_options.py`.

The values come from an API caller's `script_args` (any JSON type), from a pasted infotext or from Gradio. Checked here:
every value turns into a member of the right enum (never an exception, never a member of the other enum), only the exact
names (case and surrounding spaces aside) pick a non-default member, the list of installed backends never breaks the UI,
and the run never gets a backend the vendored pipeline could not import.
"""
import math
from enum import Enum

import pytest

from seedvr2_neo.pipeline_options import ATTENTION_PACKAGES, DEFAULT_ATTENTION_MODE, DEFAULT_COLOR_CORRECTION, \
    AttentionMode, ColorCorrection, choice_values, installed_attention_modes, parse_attention_mode, \
    parse_color_correction, resolve_attention_mode


class ForeignStrEnum(str, Enum):
    """Another str enum whose values happen to be valid names: a str subclass, so its value is what counts."""

    HSV = "hsv"
    SAGE = "sageattn_2"


class ForeignPlainEnum(Enum):
    """A non-str enum with valid names as values: not a string, so it means nothing."""

    HSV = "hsv"
    SAGE = "sageattn_2"


GARBAGE = [None, "", " ", "\t\n", "lab\x00", "la b", "l-a-b", "lab, Steps: 999", "lab\nSteps: 999", "💥" * 5000,
           "lab" * 200_000, "x" * 1_000_000, b"lab", bytearray(b"hsv"), b"sdpa", memoryview(b"sdpa"), 0, 1, -1, 3.5,
           float("nan"), float("inf"), float("-inf"), 10 ** 400, True, False, [], ["lab"], ("sdpa",), {"lab": 1},
           {"value": "hsv"}, set(), object(), complex(1, 2), ForeignPlainEnum.HSV, ForeignPlainEnum.SAGE,
           "ｌａｂ", "ｓｄｐａ", "lab​", "‮sdpa", "sdpa﻿"]


def short_id(value):
    return repr(value)[:30]


# --- parsing ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_color_correction_is_lab(value):
    assert parse_color_correction(value) is DEFAULT_COLOR_CORRECTION


@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_attention_is_sdpa(value):
    assert parse_attention_mode(value) is DEFAULT_ATTENTION_MODE


@pytest.mark.parametrize("member", list(ColorCorrection))
def test_an_attention_parse_of_a_color_member_is_sdpa_not_a_color(member):
    assert parse_attention_mode(member) is DEFAULT_ATTENTION_MODE


@pytest.mark.parametrize("member", list(AttentionMode))
def test_a_color_parse_of_an_attention_member_is_lab_not_an_attention(member):
    assert parse_color_correction(member) is DEFAULT_COLOR_CORRECTION


@pytest.mark.parametrize("member", list(ColorCorrection))
@pytest.mark.parametrize("dress", [str.upper, str.title, lambda s: f"  {s}\t", lambda s: f"\n{s.upper()} \r\n"])
def test_every_color_name_is_read_in_any_case_and_padding(member, dress):
    assert parse_color_correction(dress(member.value)) is member


@pytest.mark.parametrize("member", list(AttentionMode))
@pytest.mark.parametrize("dress", [str.upper, str.title, lambda s: f"  {s}\t", lambda s: f"\n{s.upper()} \r\n"])
def test_every_attention_name_is_read_in_any_case_and_padding(member, dress):
    assert parse_attention_mode(dress(member.value)) is member


def test_the_string_none_is_no_color_correction_but_json_null_is_lab():
    """`null` from JSON means "not sent" (the default); the string names the method that turns correction off."""
    assert parse_color_correction(None) is ColorCorrection.LAB
    assert parse_color_correction("None") is ColorCorrection.NONE
    assert parse_color_correction("NONE ") is ColorCorrection.NONE


def test_a_foreign_str_enum_counts_by_its_value_and_a_plain_enum_does_not():
    assert parse_color_correction(ForeignStrEnum.HSV) is ColorCorrection.HSV
    assert parse_attention_mode(ForeignStrEnum.SAGE) is AttentionMode.SAGEATTN_2
    assert parse_color_correction(ForeignPlainEnum.HSV) is ColorCorrection.LAB
    assert parse_attention_mode(ForeignPlainEnum.SAGE) is AttentionMode.SDPA


@pytest.mark.parametrize("value", ["ſdpa", "hſv", "LAB", "Ｌab", "wavelet_adaptive ", "wavelet adaptive", "wavelet-adaptive",
                                   "flash_attn2", "sageattn", "sage_attn_2", "flash-attn-2", "sdpa "])
def test_near_names_pick_only_an_exact_casefolded_match(value):
    """Unicode casefolding may turn a look-alike into an ASCII name (the long s becomes "s"); whatever is picked must
    be the member whose value equals the casefolded, stripped string, and anything else is the default."""
    wanted = value.strip().casefold()
    color = parse_color_correction(value)
    attention = parse_attention_mode(value)
    assert color.value == wanted or color is DEFAULT_COLOR_CORRECTION
    assert attention.value == wanted or attention is DEFAULT_ATTENTION_MODE


def test_a_huge_padded_valid_name_is_still_read():
    assert parse_attention_mode(" " * 1_000_000 + "SAGEATTN_3" + "\n" * 1_000_000) is AttentionMode.SAGEATTN_3


def test_a_str_subclass_is_read_as_a_string():
    class Tagged(str):
        pass

    assert parse_color_correction(Tagged(" Wavelet ")) is ColorCorrection.WAVELET


def test_choice_values_are_unique_strings_that_parse_back():
    for enum, parse in ((ColorCorrection, parse_color_correction), (AttentionMode, parse_attention_mode)):
        values = choice_values(enum)
        assert len(values) == len(set(values)) and all(isinstance(value, str) for value in values)
        assert [parse(value) for value in values] == list(enum)
    assert choice_values([]) == []


# --- the installed backends (the UI's list) ----------------------------------------------------------------------------

def test_nothing_installed_lists_only_sdpa():
    assert installed_attention_modes(lambda name: None) == [AttentionMode.SDPA]


def test_everything_installed_lists_every_backend_once_in_enum_order():
    assert installed_attention_modes(lambda name: object()) == list(AttentionMode)


@pytest.mark.parametrize("error", [ImportError("broken parent"), ModuleNotFoundError("gone"),
                                   ValueError("__spec__ is None")])
def test_a_find_spec_import_or_value_error_hides_only_that_backend(error):
    def find_spec(name):
        if name == "flash_attn":
            raise error
        return object()

    modes = installed_attention_modes(find_spec)
    assert AttentionMode.FLASH_ATTN_2 not in modes and len(modes) == len(AttentionMode) - 1


@pytest.mark.parametrize("spec", [0, "", [], False])
def test_a_falsy_but_not_none_spec_counts_as_installed(spec):
    """Only None means "not found" (`importlib.util.find_spec`'s contract)."""
    assert installed_attention_modes(lambda name: spec) == list(AttentionMode)


def test_find_spec_is_asked_only_for_the_known_top_level_packages():
    asked = []
    installed_attention_modes(lambda name: asked.append(name))
    assert sorted(asked) == sorted(ATTENTION_PACKAGES.values())
    assert all("." not in name and name for name in asked)


def test_the_default_find_spec_survives_a_shadowing_module_without_a_spec(monkeypatch):
    """A module object put into `sys.modules` by another extension, with `__spec__ = None`: importlib raises
    ValueError, which counts as not installed."""
    import sys
    import types

    fake = types.ModuleType("sageattention")
    fake.__spec__ = None
    monkeypatch.setitem(sys.modules, "sageattention", fake)
    modes = installed_attention_modes()
    assert AttentionMode.SAGEATTN_2 not in modes and modes[0] is AttentionMode.SDPA


@pytest.mark.parametrize("error", [RuntimeError("finder broke"), AttributeError("finder broke"),
                                   TypeError("finder broke")])
def test_any_find_spec_failure_never_breaks_the_ui_list(error):
    def find_spec(name):
        raise error

    assert installed_attention_modes(find_spec) == [AttentionMode.SDPA]


# --- the backend a run uses ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("requested", list(AttentionMode))
@pytest.mark.parametrize("available", [{}, {mode: False for mode in AttentionMode}, {AttentionMode.SDPA: False}])
def test_nothing_imported_means_sdpa_with_a_note_for_any_other_request(requested, available):
    used, note = resolve_attention_mode(requested, available)
    assert used is AttentionMode.SDPA
    assert (note is None) == (requested is AttentionMode.SDPA)
    if note:
        assert requested.value in note and "sdpa" in note


@pytest.mark.parametrize("requested", list(AttentionMode))
def test_everything_imported_keeps_the_request_without_a_note(requested):
    assert resolve_attention_mode(requested, {mode: True for mode in AttentionMode}) == (requested, None)


def test_string_keys_work_like_enum_keys():
    """The modes are str enums, so a mapping keyed by plain names finds them too."""
    assert resolve_attention_mode(AttentionMode.SAGEATTN_2, {"sageattn_2": True}) == (AttentionMode.SAGEATTN_2, None)
    assert resolve_attention_mode(AttentionMode.SAGEATTN_2, {"SAGEATTN_2": True})[0] is AttentionMode.SDPA


@pytest.mark.parametrize("flag", [None, 0, 0.0, "", [], float("nan")])
def test_falsy_or_odd_flags_never_pick_a_backend_except_nan(flag):
    used, _ = resolve_attention_mode(AttentionMode.FLASH_ATTN_3, {AttentionMode.FLASH_ATTN_3: flag})
    # NaN is truthy in Python; the vendored flags are real bools (`attention_availability` applies bool()), so the
    # mapping never holds NaN in a run.
    expected = AttentionMode.FLASH_ATTN_3 if isinstance(flag, float) and math.isnan(flag) else AttentionMode.SDPA
    assert used is expected


def test_the_note_names_only_the_enum_value_not_caller_text():
    _, note = resolve_attention_mode(AttentionMode.SAGEATTN_3, {})
    assert "\n" not in note and len(note) < 200


# --- the version 3 to version 2 step-down ------------------------------------------------------------------------------

@pytest.mark.parametrize("requested, available, expected", [
    (AttentionMode.FLASH_ATTN_3, {AttentionMode.FLASH_ATTN_2: True}, AttentionMode.FLASH_ATTN_2),
    (AttentionMode.SAGEATTN_3, {AttentionMode.SAGEATTN_2: True}, AttentionMode.SAGEATTN_2),
    (AttentionMode.FLASH_ATTN_3, {AttentionMode.SAGEATTN_2: True}, AttentionMode.SDPA),
    (AttentionMode.SAGEATTN_3, {AttentionMode.FLASH_ATTN_2: True, AttentionMode.FLASH_ATTN_3: True}, AttentionMode.SDPA),
    (AttentionMode.FLASH_ATTN_2, {AttentionMode.FLASH_ATTN_3: True}, AttentionMode.SDPA),
    (AttentionMode.SAGEATTN_2, {AttentionMode.SAGEATTN_3: True}, AttentionMode.SDPA),
    (AttentionMode.FLASH_ATTN_3, {AttentionMode.FLASH_ATTN_2: 0}, AttentionMode.SDPA),
    (AttentionMode.SAGEATTN_3, {"sageattn_2": True}, AttentionMode.SAGEATTN_2),
])
def test_a_missing_version_3_steps_down_only_within_its_family(requested, available, expected):
    used, note = resolve_attention_mode(requested, available)
    assert used is expected
    assert note is not None and requested.value in note and f"'{expected.value}'" in note


def test_the_step_down_never_loops_or_lands_on_a_missing_backend():
    for requested in AttentionMode:
        for mask in range(1 << len(AttentionMode)):
            available = {mode: bool(mask >> index & 1) for index, mode in enumerate(AttentionMode)}
            used, _ = resolve_attention_mode(requested, available)
            assert used is AttentionMode.SDPA or available[used], (requested, available)

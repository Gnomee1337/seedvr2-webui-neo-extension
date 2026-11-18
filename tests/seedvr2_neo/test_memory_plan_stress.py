"""Hostile UI values and checkpoint files for `seedvr2_neo/memory_plan.py`.

The values come from Gradio (Radio / Slider / Checkbox), from `ui-config.json` and from API `script_args` (raw JSON), so
any type can arrive. The plan must never raise and must never turn BlockSwap on without an offload device.
"""
import builtins
import itertools
import json
import os
import struct
import sys
import types
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction

import pytest

from seedvr2_neo import memory_plan as mp
from test_memory_plan import write_safetensors_header

GIB = 1024 ** 3
DIT_SIZE_MODULE = "seedvr2_vendor.utils.dit_size"


# --- "After run" ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("after_run", [
    None, "", " ", "\t\n", "keep in ram", "KEEP IN RAM", "Keep in RAM ", " Keep in RAM", "Keep  in RAM",
    "Keep in RAM\0", "Keep in RAM 🧠", "Keep in RAM" * 10_000, "unload seedvr2", 0, 1, True, False, 3.5,
    ["Keep in RAM"], {"value": "Keep in RAM"}, ("Keep in RAM",), b"Keep in RAM", object(),
    mp.AfterRun.UNLOAD, mp.AfterRun.UNLOAD.value,
])
def test_anything_but_the_exact_keep_label_unloads(after_run):
    plan = mp.build_memory_plan(after_run, 0, False)

    assert plan.cache_models is False


@pytest.mark.parametrize("after_run", [mp.AfterRun.KEEP_IN_RAM, mp.AfterRun.KEEP_IN_RAM.value, "Keep in RAM"])
def test_exact_keep_label_or_member_keeps(after_run):
    assert mp.build_memory_plan(after_run, 0, False).cache_models is True


# --- BlockSwap slider ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (True, 1), (False, 0),                                  # bool is an int subclass
    (10 ** 1000, 36), (-(10 ** 1000), 0),                    # huge ints from JSON
    (1e308, 36), (-1e308, 0), (float("-inf"), 0),
    (35.999, 35), (36.0, 36), (0.999, 0), (-0.5, 0),
    (Decimal("17.9"), 17), (Fraction(73, 2), 36),
    (" 8 ", 8), ("+8", 8), ("-8", 0), ("٣", 3),           # int() accepts Unicode digits
    ("8.0", 0), ("1e3", 0), ("0x10", 0), ("🧱", 0), ("9" * 5000, 0),
    (b"8", 8), (bytearray(b"12"), 12),
    (1j, 0), ([8], 0), ({"value": 8}, 0), ((8,), 0), (object(), 0),
])
def test_blocks_value_of_any_type_gives_an_int_in_range(value, expected):
    blocks = mp.clamp_blocks_to_swap(value)

    assert blocks == expected
    assert type(blocks) is int
    assert 0 <= blocks <= mp.MAX_BLOCKS_TO_SWAP


# --- I/O swap checkbox ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, False), (False, False), (True, True), (0, False), (1, True), (0.0, False), ("", False), ([], False),
])
def test_swap_io_values_that_bool_reads_correctly(value, expected):
    assert mp.build_memory_plan(mp.AfterRun.UNLOAD.value, 0, value).swap_io_components is expected


@pytest.mark.parametrize("value", ["False", "false", "0", "off", "no"])
def test_swap_io_false_strings_mean_off(value):
    plan = mp.build_memory_plan(mp.AfterRun.UNLOAD.value, 0, value)

    assert plan.swap_io_components is False
    assert plan.dit_offload_device is None


# --- Plan invariants over a grid of hostile combinations ---------------------------------------------------------

HOSTILE_AFTER_RUN = [None, "", "Keep in RAM", mp.AfterRun.UNLOAD.value, 42]
HOSTILE_BLOCKS = [None, -1, 0, 1, 36, 37, 10 ** 30, "abc", "16", float("nan"), float("inf"), 16.7, True]
HOSTILE_SWAP_IO = [None, False, True, 0, 1, "", "yes"]


@pytest.mark.parametrize("after_run, blocks, swap_io",
                         list(itertools.product(HOSTILE_AFTER_RUN, HOSTILE_BLOCKS, HOSTILE_SWAP_IO)))
def test_plan_invariants_hold_for_hostile_combinations(after_run, blocks, swap_io):
    plan = mp.build_memory_plan(after_run, blocks, swap_io)
    config = plan.block_swap_config()

    assert type(plan.cache_models) is bool
    assert type(plan.blocks_to_swap) is int and 0 <= plan.blocks_to_swap <= mp.MAX_BLOCKS_TO_SWAP
    assert type(plan.swap_io_components) is bool
    # upstream raises when BlockSwap is on without an offload device different from the GPU
    blockswap_on = plan.blocks_to_swap > 0 or plan.swap_io_components
    assert (plan.dit_offload_device == mp.CPU_DEVICE) is blockswap_on
    assert (plan.dit_offload_device is None) is (not blockswap_on)
    assert plan.vae_offload_device == plan.dit_offload_device
    assert config == {"blocks_to_swap": plan.blocks_to_swap, "swap_io_components": plan.swap_io_components,
                      "offload_device": plan.dit_offload_device}


def test_plan_is_frozen_and_hands_out_a_fresh_config_each_time():
    plan = mp.build_memory_plan(mp.AfterRun.UNLOAD.value, 20, True)

    with pytest.raises(FrozenInstanceError):
        plan.blocks_to_swap = 0  # type: ignore[misc]
    first = plan.block_swap_config()
    first["blocks_to_swap"] = 0  # the pipeline may mutate the dict it gets
    first["offload_device"] = "cuda:0"

    assert plan.block_swap_config() == {"blocks_to_swap": 20, "swap_io_components": True, "offload_device": "cpu"}


# --- Low-VRAM note -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("total, shows", [
    (8589410304, True),               # what an 8 GB card reports (slightly under 8 GiB)
    (10737418240, True),              # 10 GiB
    (11811160064, False),             # what a 12 GB card reports (~11 GiB)
    (int(10.9 * GIB), True),          # "11 GB" cards (2080 Ti, 1080 Ti) report under 11 GiB
    (1, True),                        # absurdly small but known
    (10 ** 30, False),                # absurdly big
])
def test_note_on_real_and_extreme_gpu_sizes(total, shows):
    warning = mp.low_vram_warning(True, total)

    assert (warning is not None) is shows
    if shows:
        assert "GB GPU" in warning and str(mp.RECOMMENDED_7B_BLOCKS_TO_SWAP) in warning


@pytest.mark.parametrize("total", [8 * GIB, 1, 10 ** 30, None, 0, -GIB])
@pytest.mark.parametrize("is_7b", [False, 0, None, ""])
def test_not_7b_never_gets_a_note(is_7b, total):
    assert mp.low_vram_warning(is_7b, total) is None


def test_note_size_text_just_below_threshold_is_readable():
    warning = mp.low_vram_warning(True, int(mp.LOW_VRAM_7B_THRESHOLD_GIB * GIB) - 1)

    # rounds to the threshold itself; cosmetic, but must not crash or print scientific notation
    assert "11.0 GB" in warning


# --- is_7b_checkpoint on hostile files ----------------------------------------------------------------------------

def _write(path, content: bytes) -> str:
    path.write_bytes(content)
    return str(path)


def _safetensors(header) -> bytes:
    data = json.dumps(header).encode()
    return struct.pack("<Q", len(data)) + data


CORRUPT_SAFETENSORS = [
    b"",
    b"\x01\x02\x03",
    struct.pack("<Q", 2 ** 64 - 1) + b"{}",
    struct.pack("<Q", 10 ** 6) + b'{"blocks.35.w": 1}',            # declared length past EOF
    struct.pack("<Q", 18) + b'{"blocks.35.w": 1',                 # truncated JSON
    struct.pack("<Q", 6) + b"\xff\xfe\xfd\xfc\xfb\xfa",
    struct.pack("<Q", 2) + b"[]",
    struct.pack("<Q", 4) + b"null",
    struct.pack("<Q", 2) + b"{}",
    _safetensors({"__metadata__": {"blocks.35.x": "1"}}),                # blocks only inside metadata
]


@pytest.mark.parametrize("content", CORRUPT_SAFETENSORS)
@pytest.mark.parametrize("name, expected", [("seedvr2_ema_7b_fp8.safetensors", True),
                                            ("seedvr2_ema_3b_fp8.safetensors", False),
                                            ("renamed.safetensors", False)])
def test_corrupt_header_falls_back_to_the_name(tmp_path, content, name, expected):
    path = _write(tmp_path / name, content)

    assert mp.is_7b_checkpoint(name, path) is expected


@pytest.mark.parametrize("block_count", [0, 1, 31, 33, 35, 37, 100])
def test_unknown_block_count_falls_back_to_the_name(tmp_path, block_count):
    path = write_safetensors_header(tmp_path / "x.safetensors", block_count)

    assert mp.is_7b_checkpoint("model_7B.safetensors", path) is True
    assert mp.is_7b_checkpoint("model.safetensors", path) is False


def test_directory_in_place_of_the_checkpoint_falls_back_to_the_name(tmp_path):
    folder = tmp_path / "seedvr2_ema_7b.safetensors"
    folder.mkdir()

    assert mp.is_7b_checkpoint(folder.name, str(folder)) is True
    assert mp.is_7b_checkpoint("seedvr2_ema_3b.safetensors", str(folder)) is False


@pytest.mark.parametrize("folder_name", ["Модели", "モデル", "models 🧠 ✨", "a b%20c", "con_ prn"])
def test_unicode_and_odd_folders_are_read(tmp_path, folder_name):
    folder = tmp_path / folder_name
    folder.mkdir()
    path = write_safetensors_header(folder / "renamed 💾.safetensors", 36)

    assert mp.is_7b_checkpoint("renamed 💾.safetensors", path) is True


def test_very_long_path_is_read_or_falls_back_without_raising(tmp_path):
    folder = tmp_path
    for _ in range(12):
        folder = folder / ("deep_" + "x" * 20)
    try:
        os.makedirs(folder, exist_ok=True)
        path = write_safetensors_header(folder / "renamed.safetensors", 36)
    except OSError:
        pytest.skip("the OS cannot create paths this long here")

    assert len(path) > 260
    assert mp.is_7b_checkpoint("renamed.safetensors", path) is True


@pytest.mark.parametrize("checkpoint_path", ["", "\0", "a\0b.safetensors", "::::", "\\\\?\\nope\\x.safetensors"])
def test_garbage_checkpoint_paths_fall_back_to_the_name(checkpoint_path):
    assert mp.is_7b_checkpoint("seedvr2_ema_7b.safetensors", checkpoint_path) is True
    assert mp.is_7b_checkpoint("seedvr2_ema_3b.safetensors", checkpoint_path) is False


def test_gguf_name_with_safetensors_content_and_no_gguf_package_falls_back(tmp_path, monkeypatch):
    path = write_safetensors_header(tmp_path / "seedvr2_ema_7b-Q4_K_M.gguf", 32)
    monkeypatch.setitem(sys.modules, "gguf", None)  # Forge without gguf: the lazy import raises ImportError

    assert mp.is_7b_checkpoint("seedvr2_ema_7b-Q4_K_M.gguf", path) is True


# --- the vendored size check is missing or foreign -----------------------------------------------------------------

def test_module_not_found_means_no_note(tmp_path, monkeypatch):
    path = write_safetensors_header(tmp_path / "seedvr2_ema_7b.safetensors", 36)
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == DIT_SIZE_MODULE:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert mp.is_7b_checkpoint("seedvr2_ema_7b.safetensors", path) is False


def test_shadowed_module_without_the_names_means_no_note(tmp_path, monkeypatch):
    path = write_safetensors_header(tmp_path / "seedvr2_ema_7b.safetensors", 36)
    monkeypatch.setitem(sys.modules, DIT_SIZE_MODULE, types.ModuleType(DIT_SIZE_MODULE))

    assert mp.is_7b_checkpoint("seedvr2_ema_7b.safetensors", path) is False


def _broken_import(monkeypatch, error):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == DIT_SIZE_MODULE:
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)


def _foreign_module(monkeypatch, error):
    module = types.ModuleType(DIT_SIZE_MODULE)

    class DitSize:
        B7 = "7B"

    def choose_dit_size(*args, **kwargs):
        raise error

    module.DitSize, module.choose_dit_size = DitSize, choose_dit_size
    monkeypatch.setitem(sys.modules, DIT_SIZE_MODULE, module)


@pytest.mark.parametrize("break_it", [
    lambda mp_: _broken_import(mp_, SyntaxError("invalid syntax (dit_size.py, line 1)")),
    lambda mp_: _broken_import(mp_, OSError("[WinError 5] Access is denied")),
    lambda mp_: _foreign_module(mp_, TypeError("choose_dit_size() takes 1 positional argument but 2 were given")),
], ids=["syntax-error-on-import", "os-error-on-import", "foreign-api"])
def test_any_failure_of_the_size_check_means_no_note(tmp_path, monkeypatch, break_it):
    path = write_safetensors_header(tmp_path / "seedvr2_ema_7b.safetensors", 36)
    break_it(monkeypatch)

    assert mp.is_7b_checkpoint("seedvr2_ema_7b.safetensors", path) is False

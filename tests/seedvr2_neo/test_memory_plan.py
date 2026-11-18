"""UI values → offload devices and cache flags, and the 7B low-VRAM note of `seedvr2_neo/memory_plan.py`."""
import json
import struct

import pytest

from seedvr2_neo import memory_plan as mp

GIB = 1024 ** 3


def write_safetensors_header(path, block_count):
    """A safetensors file holding only a header with `block_count` transformer blocks (no weights needed)."""
    header = {f"blocks.{i}.attn.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
              for i in range(block_count)}
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data + b"\0\0")
    return str(path)


def test_7b_checkpoint_is_detected_from_its_header(tmp_path):
    # renamed on purpose: the header decides, as when the pipeline picks its config
    path = write_safetensors_header(tmp_path / "renamed_model.safetensors", 36)

    assert mp.is_7b_checkpoint("renamed_model.safetensors", path) is True


def test_3b_checkpoint_is_not_7b_even_with_7b_in_the_name(tmp_path):
    path = write_safetensors_header(tmp_path / "seedvr2_ema_7b_fp16.safetensors", 32)

    assert mp.is_7b_checkpoint("seedvr2_ema_7b_fp16.safetensors", path) is False


def test_unreadable_checkpoint_falls_back_to_the_file_name(tmp_path):
    assert mp.is_7b_checkpoint("seedvr2_ema_7B_fp8.safetensors", str(tmp_path / "missing")) is True
    assert mp.is_7b_checkpoint("seedvr2_ema_3b_fp8.safetensors", str(tmp_path / "missing")) is False


def test_unload_without_blockswap_caches_nothing_and_offloads_nothing():
    plan = mp.build_memory_plan(mp.AfterRun.UNLOAD.value, 0, False)

    assert plan.cache_models is False
    assert plan.dit_offload_device is None
    assert plan.vae_offload_device is None
    assert plan.block_swap_config() == {"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None}


def test_keep_in_ram_caches_models():
    plan = mp.build_memory_plan(mp.AfterRun.KEEP_IN_RAM.value, 0, False)

    assert plan.cache_models is True


def test_blockswap_sets_cpu_offload_for_dit_and_vae():
    plan = mp.build_memory_plan(mp.AfterRun.UNLOAD.value, 20, False)

    assert plan.dit_offload_device == mp.CPU_DEVICE
    assert plan.vae_offload_device == mp.CPU_DEVICE
    assert plan.block_swap_config() == {"blocks_to_swap": 20, "swap_io_components": False, "offload_device": "cpu"}


def test_io_swap_alone_also_needs_an_offload_device():
    plan = mp.build_memory_plan(mp.AfterRun.UNLOAD.value, 0, True)

    assert plan.dit_offload_device == mp.CPU_DEVICE
    assert plan.block_swap_config()["swap_io_components"] is True


def test_unknown_after_run_value_unloads():
    assert mp.build_memory_plan("something from an old ui-config", 0, False).cache_models is False


@pytest.mark.parametrize("value, expected", [
    (0, 0), (1, 1), (36, 36), (37, 36), (10 ** 9, 36), (-5, 0), (12.9, 12), ("8", 8),
    (None, 0), ("", 0), ("abc", 0), (float("inf"), 0), (float("nan"), 0),
])
def test_blocks_to_swap_is_clamped(value, expected):
    assert mp.clamp_blocks_to_swap(value) == expected


def test_garbage_blocks_value_means_blockswap_off():
    plan = mp.build_memory_plan(mp.AfterRun.UNLOAD.value, "abc", False)

    assert plan.blocks_to_swap == 0
    assert plan.dit_offload_device is None


def test_7b_on_8_gb_gpu_gets_a_warning():
    warning = mp.low_vram_warning(True, 8 * GIB)

    assert warning is not None
    assert "8.0 GB" in warning
    assert "BlockSwap" in warning


@pytest.mark.parametrize("is_7b, total", [
    (False, 8 * GIB),         # 3B fits
    (True, 24 * GIB),         # big GPU
    (True, 12 * GIB),         # at the threshold or above: no note
    (True, None),             # GPU size unknown (CPU, MPS, read error)
    (True, 0),
    (True, -1),
])
def test_no_warning(is_7b, total):
    assert mp.low_vram_warning(is_7b, total) is None


def test_threshold_boundary():
    threshold = int(mp.LOW_VRAM_7B_THRESHOLD_GIB * GIB)

    assert mp.low_vram_warning(True, threshold) is None
    assert mp.low_vram_warning(True, threshold - 1) is not None

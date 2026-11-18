"""UI values → offload devices and cache flags, and the 7B low-VRAM note of `seedvr2_neo/memory_plan.py`."""
import pytest

from seedvr2_neo import memory_plan as mp

GIB = 1024 ** 3


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
    warning = mp.low_vram_warning("7b", 8 * GIB)

    assert warning is not None
    assert "8.0 GB" in warning
    assert "BlockSwap" in warning


@pytest.mark.parametrize("dit_size, total", [
    ("3b", 8 * GIB),          # 3B fits
    ("7b", 24 * GIB),         # big GPU
    ("7b", 12 * GIB),         # at the threshold or above: no note
    ("7b", None),             # GPU size unknown (CPU, MPS, read error)
    ("7b", 0),
    ("7b", -1),
    (None, 8 * GIB),          # model size unknown
])
def test_no_warning(dit_size, total):
    assert mp.low_vram_warning(dit_size, total) is None


def test_threshold_boundary():
    threshold = int(mp.LOW_VRAM_7B_THRESHOLD_GIB * GIB)

    assert mp.low_vram_warning("7b", threshold) is None
    assert mp.low_vram_warning("7b", threshold - 1) is not None

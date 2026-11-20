"""Hostile batches for the per-size passes — `seedvr2_neo/upscale_run.py` with `split_into_passes` of
`seedvr2_neo/run_inputs.py`.

In the scale factor and long edge modes every input size has its own output size, so a batch of mixed sizes (the img2img
Batch tab, an API call with many `init_images`) is planned size by size. Checked here: every input is upscaled once,
every pass holds one size and gets that size's target, the pass budget holds per size, and nothing of one run leaks into
the next run with another mode.
"""
import random
from types import SimpleNamespace
from typing import List, Tuple

import pytest
from PIL import Image

from seedvr2_neo import run_inputs as ri
from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import build_memory_plan
from seedvr2_neo.model_lifecycle import ModelLifecycle
from seedvr2_neo.upscale_run import HostHooks, UpscaleRequest, UpscaleRun, describe_target
from seedvr2_neo.upscale_settings import MAX_OUTPUT_LONG_EDGE, SizeMode, parse_upscale_settings


class Cache:
    def remove_dit(self, config):
        return False

    def remove_vae(self, config):
        return False


class RecordingPipeline(SimpleNamespace):
    """Phases that return the input frames; records the input size and target of every pass."""

    def __init__(self):
        super().__init__(Debug=lambda enabled: None, get_global_cache=Cache, passes=[])

    def setup_generation_context(self, **kwargs):
        return {}

    def prepare_runner(self, **kwargs):
        return None, None

    def compute_generation_info(self, ctx, images, resolution, max_resolution, **kwargs):
        count, height, width, _ = images.shape
        self.passes.append({"size": (width, height), "count": count, "resolution": resolution,
                            "max_resolution": max_resolution})
        return images, {}

    def encode_all_batches(self, runner, ctx, images, progress_callback, **kwargs):
        progress_callback()
        return {"frames": images}

    def upscale_all_batches(self, runner, ctx, progress_callback, **kwargs):
        return ctx

    def decode_all_batches(self, runner, ctx, progress_callback, **kwargs):
        return ctx

    def postprocess_all_batches(self, ctx, progress_callback, **kwargs):
        return {"final_video": ctx["frames"]}


def run_batch(images, mode, scale_factor=2.0, long_edge=2048, resolution=1080, tmp_path="models"):
    settings = parse_upscale_settings(resolution, 0.0, 0.0, True, 512, 128, mode, scale_factor, long_edge)
    pipeline = RecordingPipeline()
    hooks = HostHooks(device="gpu-0", release_memory=lambda: None, is_interrupted=lambda: False)
    run = UpscaleRun(pipeline, ModelLifecycle(LoadedModelTracker(), lambda: None), hooks)
    request = UpscaleRequest("dit.gguf", "vae.safetensors", str(tmp_path), build_memory_plan(None, 0, False), 7,
                             settings)
    return run.upscale_images(images, request, False, False), pipeline, settings


def mixed_batch(sizes: List[Tuple[int, int]], count: int, seed: int = 3) -> List[Image.Image]:
    rng = random.Random(seed)
    return [Image.new("RGB", rng.choice(sizes)) for _ in range(count)]


# --- mixed sizes in one run ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(SizeMode))
def test_every_input_of_a_mixed_batch_is_upscaled_once_with_its_own_target(mode):
    sizes = [(40, 30), (30, 40), (64, 64), (16, 64), (64, 16), (3, 5), (1, 1)]
    images = mixed_batch(sizes, 150)
    outcome, pipeline, settings = run_batch(images, mode)
    assert outcome.failure is None and sorted(outcome.upscaled) == list(range(len(images)))
    assert sum(entry["count"] for entry in pipeline.passes) == len(images)
    for entry in pipeline.passes:
        target = settings.target_for(entry["size"])
        assert (entry["resolution"], entry["max_resolution"]) == (target.resolution, target.max_resolution)
        if mode is not SizeMode.SHORTEST_EDGE:
            assert entry["max_resolution"] <= MAX_OUTPUT_LONG_EDGE


@pytest.mark.parametrize("mode", list(SizeMode))
def test_the_pass_budget_holds_for_every_size_of_a_mixed_batch(mode):
    """Small inputs are raised to a 512 px shortest edge, so 300 tiny frames in several sizes need several passes."""
    sizes = [(40, 30), (30, 40), (20, 80), (64, 64)]
    images = mixed_batch(sizes, 300)
    outcome, pipeline, settings = run_batch(images, mode, scale_factor=4.0, long_edge=3840, resolution=1536)
    assert outcome.failure is None
    for entry in pipeline.passes:
        pass_pixels = entry["count"] * settings.target_for(entry["size"]).output_pixels(entry["size"])
        assert entry["count"] == 1 or pass_pixels <= ri.PASS_BUDGET_PIXELS, entry
    assert len(pipeline.passes) > len(sizes)  # the budget really split the groups


def test_split_into_passes_keeps_order_size_and_budget_for_a_huge_random_batch():
    rng = random.Random(99)
    sizes = [(rng.randint(1, 4000), rng.randint(1, 4000)) for _ in range(40)]
    images = [SimpleNamespace(size=rng.choice(sizes), mode="RGB") for _ in range(5000)]
    for mode in SizeMode:
        settings = parse_upscale_settings(1080, 0, 0, True, 512, 128, mode, rng.uniform(1, 4), rng.randint(1024, 3840))
        targets = {size: settings.target_for(size) for size in sizes}
        groups = ri.group_by_shape(images)
        passes = ri.split_into_passes(groups, images, lambda size: targets[size].output_pixels(size))
        assert sorted(index for one_pass in passes for index in one_pass) == list(range(len(images)))
        for one_pass in passes:
            assert one_pass == sorted(one_pass)
            assert len({images[index].size for index in one_pass}) == 1
            pixels = sum(targets[images[index].size].output_pixels(images[index].size) for index in one_pass)
            assert len(one_pass) == 1 or pixels <= ri.PASS_BUDGET_PIXELS


def test_a_pass_never_mixes_transposed_sizes_which_get_transposed_targets():
    images = [Image.new("RGB", (40, 30)), Image.new("RGB", (30, 40))] * 3
    outcome, pipeline, settings = run_batch(images, "Scale factor", scale_factor=4.0)
    assert sorted((entry["size"], entry["count"]) for entry in pipeline.passes) == [((30, 40), 3), ((40, 30), 3)]


# --- the console line ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(SizeMode))
@pytest.mark.parametrize("size", [(1, 1), (7, 3), (512, 384), (1000, 16000), (3840, 2160), (20000, 20000)])
def test_the_console_line_names_the_planned_size(mode, size):
    settings = parse_upscale_settings(1080, 0, 0, True, 512, 128, mode, 2.0, 2048)
    target = settings.target_for(size)
    line = describe_target(size, target, 3)
    out_w, out_h = target.output_size(size)
    assert line.startswith("[SeedVR2] 3 images ") and f"{size[0]}x{size[1]} -> {out_w}x{out_h}" in line
    assert ("raised to the 512 px minimum" in line) == target.raised_to_minimum
    assert "\n" not in line


# --- runs one after another with different modes ----------------------------------------------------------------

def test_consecutive_runs_with_different_modes_do_not_share_targets():
    images = [Image.new("RGB", (40, 30)), Image.new("RGB", (64, 64))]
    plans = []
    for mode, factor, edge in [("Scale factor", 4.0, 2048), ("Long edge", 2.0, 1024), ("Shortest edge", 2.0, 2048),
                               ("Scale factor", 1.0, 2048), ("Long edge", 2.0, 3840)]:
        _, pipeline, settings = run_batch(images, mode, factor, edge, resolution=1536)
        plans.append([(entry["size"], entry["resolution"], entry["max_resolution"]) for entry in pipeline.passes])
        assert plans[-1] == [(image.size, settings.target_for(image.size).resolution,
                              settings.target_for(image.size).max_resolution) for image in images]
    assert plans[0] != plans[1] != plans[2]
    _, pipeline, _ = run_batch(images, "Scale factor", 4.0, 2048, resolution=1536)
    assert [(entry["size"], entry["resolution"], entry["max_resolution"]) for entry in pipeline.passes] == plans[0]

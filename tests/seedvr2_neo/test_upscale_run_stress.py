"""Hostile batches for the per-size passes — `seedvr2_neo/upscale_run.py` with `split_into_passes` of
`seedvr2_neo/run_inputs.py`.

In the scale factor and long edge modes every input size has its own output size, so a batch of mixed sizes (the img2img
Batch tab, an API call with many `init_images`) is planned size by size. Checked here: every input is upscaled once,
every pass holds one size and gets that size's target, the pass budget holds per size, and nothing of one run leaks into
the next run with another mode.
"""
import random
import types
from types import SimpleNamespace
from typing import List, Tuple

import pytest
import torch
from PIL import Image

from seedvr2_neo import run_inputs as ri
from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import AfterRun, build_memory_plan
from seedvr2_neo.model_lifecycle import ModelLifecycle
from seedvr2_neo.upscale_run import INTERRUPTED_MESSAGE, HostHooks, UpscaleRequest, UpscaleRun, describe_target
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
    assert ("raised to" in line) == target.raised_to_minimum
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


# --- the module hooks, Interrupt inside a phase and the progress report ---------------------------------------------
#
# A run puts forward pre-hooks on the container modules of the cached DiT and VAE. Whatever goes wrong (the report
# function raises, Interrupt fires inside a BlockSwap-wrapped block, the runner holds no modules, the models are
# swapped mid-phase), the run must end with an outcome (never an exception), no hook left on the models, and the
# models dropped unless the run succeeded.

class Boom(Exception):
    pass


class Inner(torch.nn.Module):
    """An attention-like container inside a block: its hook fires while the block's own forward runs."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Identity()

    def forward(self, x):
        return self.proj(x)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Inner()
        self.on_gpu = False

    def forward(self, x):
        return self.attn(x) + 1


class Dit(torch.nn.Module):
    def __init__(self, count=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(count))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Identity()))

    def forward(self, x):
        return self.decoder(x)


def wrap_like_blockswap(block):
    """The vendored BlockSwap wrapper (`seedvr2_vendor/optimization/blockswap.py`, `_wrap_block_forward`): move the
    block to the GPU, run it, move it back, with no try/finally, so an exception inside leaves it on the GPU."""
    original_forward = block.forward

    def wrapped_forward(self, *args, **kwargs):
        self.on_gpu = True
        output = original_forward(*args, **kwargs)
        self.on_gpu = False
        return output

    block.forward = types.MethodType(wrapped_forward, block)
    block._original_forward = original_forward


class RecordingCache(Cache):
    def __init__(self):
        self.removed = []

    def remove_dit(self, config):
        self.removed.append("dit")
        return True

    def remove_vae(self, config):
        self.removed.append("vae")
        return True


class ModelPipeline(SimpleNamespace):
    """Runs the VAE in encode and decode, the DiT in upscale; `runner` may hold anything."""

    def __init__(self, runner=None, swap_dit_in_upscale=None):
        cache = RecordingCache()
        super().__init__(Debug=lambda enabled: None, get_global_cache=lambda: cache, cache=cache)
        self.dit = Dit()
        self.vae = Vae()
        self.runner = runner if runner is not None else SimpleNamespace(dit=self.dit, vae=self.vae)
        self.swap_dit_in_upscale = swap_dit_in_upscale

    def setup_generation_context(self, **kwargs):
        return {}

    def prepare_runner(self, **kwargs):
        return self.runner, None

    def compute_generation_info(self, ctx, images, **kwargs):
        return images, {}

    @staticmethod
    def _run(model):
        if isinstance(model, torch.nn.Module):
            model(torch.zeros(1))

    def encode_all_batches(self, runner, ctx, images, progress_callback, **kwargs):
        self._run(getattr(runner, "vae", None))
        progress_callback(1, 1)
        return {"frames": images}

    def upscale_all_batches(self, runner, ctx, progress_callback, **kwargs):
        if self.swap_dit_in_upscale is not None:
            runner.dit = self.swap_dit_in_upscale
        self._run(getattr(runner, "dit", None))
        progress_callback(1, 1)
        return ctx

    def decode_all_batches(self, runner, ctx, progress_callback, **kwargs):
        self._run(getattr(runner, "vae", None))
        progress_callback(1, 1)
        return ctx

    def postprocess_all_batches(self, ctx, progress_callback, **kwargs):
        progress_callback(1, 1)
        return {"final_video": ctx["frames"]}


def hooks_on(*models):
    return sum(len(module._forward_pre_hooks) for model in models if isinstance(model, torch.nn.Module)
               for module in model.modules())


def model_run(pipeline, tmp_path, report=lambda fraction, text: None, is_interrupted=lambda: False,
              release_memory=lambda: None, after_run=AfterRun.KEEP_IN_RAM.value, images=1):
    hooks = HostHooks(device="gpu-0", release_memory=release_memory, is_interrupted=is_interrupted,
                      report_progress=report)
    run = UpscaleRun(pipeline, ModelLifecycle(LoadedModelTracker(), lambda: None), hooks)
    settings = parse_upscale_settings(1080, 0.0, 0.0, True, 512, 128)
    request = UpscaleRequest("dit.gguf", "vae.safetensors", str(tmp_path), build_memory_plan(after_run, 0, False), 7,
                             settings)
    inputs = [Image.new("RGB", (4 + index, 4)) for index in range(images)]  # one size each: one pass each
    return run, run.upscale_images(inputs, request, False, False)


def dropped_after_run(pipeline):
    """Both models removed by the run's end, beyond the drop of `prepare_run` (no remembered files in a fresh
    lifecycle, so it always drops them first)."""
    return pipeline.cache.removed == ["dit", "vae", "dit", "vae"]


def count_reports(tmp_path, images=1):
    calls = []
    model_run(ModelPipeline(), tmp_path, report=lambda fraction, text: calls.append(fraction), images=images)
    return len(calls)


@pytest.mark.parametrize("images", [1, 2])
def test_a_report_raising_at_any_point_ends_the_run_cleanly(tmp_path, images):
    """The progress bar is someone else's code; whichever call raises, no exception escapes, no hook stays on the
    cached models and the models are dropped even with "Keep in RAM"."""
    total = count_reports(tmp_path, images)
    assert total > 8
    for failing_call in range(1, total + 1):
        pipeline = ModelPipeline()
        calls = []

        def report(fraction, text):
            calls.append(fraction)
            if len(calls) == failing_call:
                raise Boom()

        run, outcome = model_run(pipeline, tmp_path, report=report, images=images)
        assert outcome.failure == "Runtime Error: Boom", failing_call
        assert hooks_on(pipeline.dit, pipeline.vae) == 0, failing_call
        assert dropped_after_run(pipeline), failing_call


def test_a_report_raising_at_the_dit_phase_start_clears_the_block_filter(tmp_path):
    def report(fraction, text):
        if text == "SeedVR2: Upscaling":
            raise Boom()

    run, outcome = model_run(ModelPipeline(), tmp_path, report=report)
    assert outcome.failure == "Runtime Error: Boom"
    assert run._dit_blocks == ()


@pytest.mark.parametrize("raiser", ["is_interrupted", "release_memory"])
def test_a_host_hook_raising_ends_the_run_cleanly(tmp_path, raiser):
    def boom():
        raise Boom()

    pipeline = ModelPipeline()
    _, outcome = model_run(pipeline, tmp_path, **{raiser: boom})
    assert outcome.failure == "Runtime Error: Boom" and outcome.upscaled == {}
    assert hooks_on(pipeline.dit, pipeline.vae) == 0
    assert dropped_after_run(pipeline)


@pytest.mark.parametrize("where", ["block", "inner"])
def test_interrupt_inside_a_blockswap_wrapped_block_drops_the_models_even_with_keep_in_ram(tmp_path, where):
    """With BlockSwap the vendored wrapper has no try/finally: Interrupt from the hook of a container inside the block
    leaves that block on the GPU. The run must then drop the models, or "Keep in RAM" would keep the block in VRAM."""
    pipeline = ModelPipeline()
    for block in pipeline.dit.blocks:
        wrap_like_blockswap(block)
    pressed = []
    target = pipeline.dit.blocks[1] if where == "block" else pipeline.dit.blocks[1].attn
    target.register_forward_pre_hook(lambda module, args: pressed.append(1))
    own_hooks = hooks_on(pipeline.dit, pipeline.vae)

    _, outcome = model_run(pipeline, tmp_path, is_interrupted=lambda: bool(pressed))

    assert outcome.failure == INTERRUPTED_MESSAGE and outcome.upscaled == {}
    assert hooks_on(pipeline.dit, pipeline.vae) == own_hooks
    # The block's hook fires before the wrapper moves the block; the inner one after, so the block stays "on the GPU".
    assert pipeline.dit.blocks[1].on_gpu == (where == "inner")
    assert not pipeline.dit.blocks[2].on_gpu
    assert dropped_after_run(pipeline)


def test_the_hooks_fire_through_a_replaced_forward(tmp_path):
    """BlockSwap replaces `forward` on the instance; module hooks still run (`Module.__call__`), so the DiT blocks
    still move the bar."""
    pipeline = ModelPipeline()
    for block in pipeline.dit.blocks:
        wrap_like_blockswap(block)
    texts = []
    _, outcome = model_run(pipeline, tmp_path, report=lambda fraction, text: texts.append(text))
    assert outcome.failure is None
    assert texts.count("SeedVR2: Upscaling") == 1 + len(pipeline.dit.blocks) + 1


@pytest.mark.parametrize("runner", [SimpleNamespace(), SimpleNamespace(dit=None, vae=None),
                                    SimpleNamespace(dit="dit", vae="vae"), SimpleNamespace(dit=3, vae=b"vae"),
                                    SimpleNamespace(dit={"blocks": [1, 2]}, vae=[torch.nn.Identity()]),
                                    SimpleNamespace(dit=SimpleNamespace(dit_model=None, blocks=[1]), vae=None),
                                    SimpleNamespace(dit=SimpleNamespace(dit_model=SimpleNamespace(blocks=iter([1, 2]))),
                                                    vae=SimpleNamespace(modules=lambda: [torch.nn.Identity()]))],
                         ids=lambda runner: repr(runner)[:40])
def test_a_runner_without_modules_runs_without_hooks(tmp_path, runner):
    pipeline = ModelPipeline(runner=runner)
    _, outcome = model_run(pipeline, tmp_path, after_run=AfterRun.UNLOAD.value)
    assert outcome.failure is None and list(outcome.upscaled) == [0]


def test_a_layer_as_the_whole_model_gets_no_hook(tmp_path):
    """A bare layer (no children) is not a container: nothing is watched, nothing is left."""
    layer = torch.nn.Linear(1, 1)
    pipeline = ModelPipeline(runner=SimpleNamespace(dit=layer, vae=layer))
    _, outcome = model_run(pipeline, tmp_path)
    assert outcome.failure is None and len(layer._forward_pre_hooks) == 0


def test_the_same_module_as_dit_and_vae_gets_every_hook_removed(tmp_path):
    """Watched twice (the list is not de-duplicated): two hooks per container, both removed; the bar stays sane."""
    dit = Dit()
    pipeline = ModelPipeline(runner=SimpleNamespace(dit=dit, vae=dit))
    reports = []
    _, outcome = model_run(pipeline, tmp_path, report=lambda fraction, text: reports.append(fraction))
    assert outcome.failure is None and hooks_on(dit) == 0
    assert all(0.0 <= fraction <= 1.0 for fraction in reports) and reports == sorted(reports)


def test_a_vae_nested_in_the_dit_is_cleaned(tmp_path):
    dit = Dit()
    dit.vae = Vae()
    pipeline = ModelPipeline(runner=SimpleNamespace(dit=dit, vae=dit.vae))
    _, outcome = model_run(pipeline, tmp_path)
    assert outcome.failure is None and hooks_on(dit) == 0


def test_a_dit_replaced_during_the_phase_leaves_no_hook_on_either(tmp_path):
    """If a loader ever swapped the DiT object mid-phase, the new one is simply not watched; the old one is cleaned."""
    new_dit = Dit(3)
    pipeline = ModelPipeline(swap_dit_in_upscale=new_dit)
    old_dit = pipeline.dit
    _, outcome = model_run(pipeline, tmp_path)
    assert outcome.failure is None
    assert hooks_on(old_dit, new_dit, pipeline.vae) == 0


class RefusingModule(torch.nn.Module):
    """A container whose hook registration fails, after other modules already got theirs."""

    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Sequential(torch.nn.Identity())

    def register_forward_pre_hook(self, *args, **kwargs):
        raise Boom()

    def forward(self, x):
        return self.inner(x)


def test_a_failing_hook_registration_removes_the_hooks_already_set(tmp_path):
    vae = Vae()
    vae.refuser = RefusingModule()  # listed after the decoder, so the decoder is hooked first
    dit = Dit()
    pipeline = ModelPipeline(runner=SimpleNamespace(dit=dit, vae=vae))
    _, outcome = model_run(pipeline, tmp_path)
    assert outcome.failure == "Runtime Error: Boom"
    assert hooks_on(vae, dit) == 0
    assert dropped_after_run(pipeline)


def test_interrupt_after_a_pass_keeps_that_pass_and_stops_before_the_next(tmp_path):
    pipeline = ModelPipeline()
    pressed = []
    original = pipeline.postprocess_all_batches

    def postprocess(ctx, progress_callback, **kwargs):
        result = original(ctx, lambda *args: None, **kwargs)
        pressed.append(1)  # Interrupt right after the first pass finished
        return result

    pipeline.postprocess_all_batches = postprocess
    _, outcome = model_run(pipeline, tmp_path, is_interrupted=lambda: bool(pressed), images=3)
    assert outcome.failure == INTERRUPTED_MESSAGE
    assert list(outcome.upscaled) == [0]
    assert hooks_on(pipeline.dit, pipeline.vae) == 0
    assert dropped_after_run(pipeline)

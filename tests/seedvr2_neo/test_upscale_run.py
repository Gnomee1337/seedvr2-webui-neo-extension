"""Running the pipeline over images — `seedvr2_neo/upscale_run.py`. The whole run through the Forge script is covered in
`tests/scripts/test_seedvr2_script.py`; here are the parts that script tests cannot reach directly."""
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import build_memory_plan
from seedvr2_neo.model_lifecycle import ModelLifecycle
from seedvr2_neo.pipeline_options import AttentionMode, ColorCorrection, PipelineOptions
from seedvr2_neo.upscale_run import INTERRUPTED_MESSAGE, HostHooks, UpscaleInterrupted, UpscaleRequest, UpscaleRun, \
    attention_availability, container_modules, dit_block_types, images_to_tensor, tensor_to_images
from seedvr2_neo.upscale_settings import parse_upscale_settings


# --- tensors ---

def test_images_become_one_float16_frame_tensor_in_0_to_1():
    tensor = images_to_tensor([Image.new("RGB", (4, 2), (255, 0, 0)), Image.new("RGB", (4, 2), (0, 0, 0))])
    assert tensor.shape == (2, 2, 4, 3) and tensor.dtype == torch.float16
    assert tensor[0, 0, 0].tolist() == [1.0, 0.0, 0.0] and tensor[1].max() == 0


@pytest.mark.parametrize("mode, channels", [("RGB", 3), ("RGBA", 4)])
def test_frames_come_back_as_images_of_the_same_mode(mode, channels):
    images = tensor_to_images(images_to_tensor([Image.new(mode, (4, 2), (255,) * channels)]))
    assert len(images) == 1 and images[0].mode == mode and images[0].size == (4, 2)


def test_out_of_range_frame_values_are_clipped():
    frames = torch.tensor([[[[-1.0, 0.5, 2.0]]]], dtype=torch.float16)
    assert tensor_to_images(frames)[0].getpixel((0, 0)) == (0, 127, 255)


# --- the run ---

class Cache:
    def remove_dit(self, config):
        return False

    def remove_vae(self, config):
        return False


class Pipeline(SimpleNamespace):
    """A pipeline whose phases return the input frames unchanged, or `output_frames` when set."""

    def __init__(self, output_frames=None):
        super().__init__(Debug=lambda enabled: None, get_global_cache=Cache, output_frames=output_frames, devices=[])

    def setup_generation_context(self, **kwargs):
        self.devices.append(kwargs["dit_device"])
        return {}

    def prepare_runner(self, **kwargs):
        return None, None

    def compute_generation_info(self, ctx, images, **kwargs):
        return images, {}

    def encode_all_batches(self, runner, ctx, images, progress_callback, **kwargs):
        progress_callback()
        return {"frames": images}

    def upscale_all_batches(self, runner, ctx, progress_callback, **kwargs):
        return ctx

    def decode_all_batches(self, runner, ctx, progress_callback, **kwargs):
        return ctx

    def postprocess_all_batches(self, ctx, progress_callback, **kwargs):
        frames = ctx["frames"] if self.output_frames is None else self.output_frames
        return {"final_video": frames}


def make_run(pipeline, interrupted=False):
    releases = []
    hooks = HostHooks(device="gpu-0", release_memory=lambda: releases.append(1), is_interrupted=lambda: interrupted)
    return UpscaleRun(pipeline, ModelLifecycle(LoadedModelTracker(), lambda: None), hooks), releases


def make_request(tmp_path):
    settings = parse_upscale_settings(1080, 0.0, 0.0, True, 512, 128)
    return UpscaleRequest("dit.gguf", "ema_vae_fp16.safetensors", str(tmp_path), build_memory_plan(None, 0, False), 7,
                          settings)


def test_the_run_uses_the_host_device_and_releases_memory_after_each_phase(tmp_path):
    pipeline = Pipeline()
    run, releases = make_run(pipeline)
    outcome = run.upscale_images([Image.new("RGB", (4, 4))], make_request(tmp_path), False, False)
    assert outcome.failure is None and list(outcome.upscaled) == [0]
    assert pipeline.devices == ["gpu-0"]
    assert len(releases) == 3  # after encode, upscale and decode


def test_a_pass_with_a_wrong_number_of_outputs_is_a_failure(tmp_path):
    pipeline = Pipeline(output_frames=torch.zeros(1, 4, 4, 3))
    run, _ = make_run(pipeline)
    outcome = run.upscale_images([Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))], make_request(tmp_path), False,
                                 False)
    assert outcome.upscaled == {} and outcome.failure == "Runtime Error: RuntimeError"


def test_an_interrupt_before_a_pass_stops_the_run(tmp_path):
    run, _ = make_run(Pipeline(), interrupted=True)
    outcome = run.upscale_images([Image.new("RGB", (4, 4))], make_request(tmp_path), False, False)
    assert outcome.upscaled == {} and outcome.failure == INTERRUPTED_MESSAGE


def test_the_interrupt_check_raises_only_when_interrupted():
    make_run(Pipeline())[0].check_interruption()
    with pytest.raises(UpscaleInterrupted):
        make_run(Pipeline(), interrupted=True)[0].check_interruption()


# --- options, Interrupt inside a phase, progress ---

class Block(torch.nn.Module):
    """A transformer block: like the vendored ones, it holds layers of its own."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, x):
        return self.layer(x) + 1


class Vae(torch.nn.Module):
    """A VAE whose decoder runs `tiles` times per decode, each through two stages."""

    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Identity()),
                                           torch.nn.Sequential(torch.nn.Identity()))

    def decode(self, tiles):
        for _ in range(tiles):
            self.decoder(torch.zeros(1))


class Dit(torch.nn.Module):
    """A DiT with `count` blocks, like the vendored NaDiT's `blocks`."""

    def __init__(self, count):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(count))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ModelPipeline(Pipeline):
    """Runs a real torch DiT in the upscale phase and a small module in the decode phase, so the module hook fires;
    `on_block` runs before each DiT block (e.g. to press Interrupt)."""

    def __init__(self, blocks=4, wrapped=False, on_block=None):
        super().__init__()
        self.dit = Dit(blocks)
        self.vae = Vae()
        self.runner = SimpleNamespace(dit=SimpleNamespace(dit_model=self.dit) if wrapped else self.dit, vae=self.vae)
        self.on_decode = None  # runs before each decoded tile
        if on_block is not None:
            for block in self.dit.blocks:
                block.register_forward_pre_hook(lambda module, args: on_block())
        self.contexts = []
        self.runner_kwargs = []
        self.postprocess_kwargs = []

    def prepare_runner(self, **kwargs):
        self.runner_kwargs.append(kwargs)
        return self.runner, None

    def encode_all_batches(self, runner, ctx, images, progress_callback, **kwargs):
        self.contexts.append(ctx)
        return super().encode_all_batches(runner, ctx, images, progress_callback, **kwargs)

    def upscale_all_batches(self, runner, ctx, progress_callback, **kwargs):
        self.dit(torch.zeros(1))
        progress_callback(1, 1, 1, "upscale")
        return ctx

    def decode_all_batches(self, runner, ctx, progress_callback, **kwargs):
        for _ in range(3):
            if self.on_decode is not None:
                self.on_decode()
            self.vae.decode(1)
        return ctx

    def postprocess_all_batches(self, ctx, progress_callback, **kwargs):
        self.postprocess_kwargs.append(kwargs)
        return super().postprocess_all_batches(ctx, progress_callback, **kwargs)


def make_tracked_run(pipeline, is_interrupted=lambda: False):
    reports = []
    hooks = HostHooks(device="gpu-0", release_memory=lambda: None, is_interrupted=is_interrupted,
                      report_progress=lambda fraction, text: reports.append((fraction, text)))
    return UpscaleRun(pipeline, ModelLifecycle(LoadedModelTracker(), lambda: None), hooks), reports


def hook_count(pipeline):
    """Forward pre-hooks on every module of the pipeline's models and torch's process-wide ones."""
    modules = list(pipeline.dit.modules()) + list(pipeline.vae.modules())
    return (sum(len(module._forward_pre_hooks) for module in modules),
            len(torch.nn.modules.module._global_forward_pre_hooks))


def one_image():
    return [Image.new("RGB", (4, 4))]


def test_the_chosen_options_reach_the_pipeline(tmp_path):
    pipeline = ModelPipeline()
    run, _ = make_tracked_run(pipeline)
    request = replace(make_request(tmp_path), options=PipelineOptions(ColorCorrection.WAVELET, AttentionMode.SAGEATTN_2))
    assert run.upscale_images(one_image(), request, False, False).failure is None
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sageattn_2"
    assert pipeline.postprocess_kwargs[0]["color_correction"] == "wavelet"


def test_the_default_options_are_lab_and_sdpa(tmp_path):
    pipeline = ModelPipeline()
    make_tracked_run(pipeline)[0].upscale_images(one_image(), make_request(tmp_path), False, False)
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sdpa"
    assert pipeline.postprocess_kwargs[0]["color_correction"] == "lab"


def test_the_pipelines_own_interrupt_check_is_the_hosts(tmp_path):
    pipeline = ModelPipeline()
    pressed = []
    run, _ = make_tracked_run(pipeline, is_interrupted=lambda: bool(pressed))
    run.upscale_images(one_image(), make_request(tmp_path), False, False)
    interrupt_fn = pipeline.contexts[0]["interrupt_fn"]
    interrupt_fn()
    pressed.append(1)
    with pytest.raises(UpscaleInterrupted):
        interrupt_fn()


def test_interrupt_inside_the_dit_stops_before_the_next_block(tmp_path):
    pressed = []
    started = []

    def press_on_second_block():
        started.append(1)
        if len(started) == 2:
            pressed.append(1)

    run, _ = make_tracked_run(ModelPipeline(blocks=6, on_block=press_on_second_block),
                              is_interrupted=lambda: bool(pressed))
    outcome = run.upscale_images(one_image(), make_request(tmp_path), False, False)
    assert outcome.failure == INTERRUPTED_MESSAGE and outcome.upscaled == {}
    assert len(started) == 2  # the third block never started


@pytest.mark.parametrize("wrapped", [False, True])
def test_dit_blocks_move_the_progress_inside_the_upscale_phase(tmp_path, wrapped):
    run, reports = make_tracked_run(ModelPipeline(blocks=4, wrapped=wrapped))
    run.upscale_images(one_image(), make_request(tmp_path), False, False)
    upscaling = [fraction for fraction, text in reports if text == "SeedVR2: Upscaling"]
    assert len(upscaling) == 1 + 4 + 1  # the phase start, each block, the batch done
    assert upscaling[:5] == sorted(set(upscaling[:5]))  # each block moves the bar forward
    assert upscaling[-1] == pytest.approx(upscaling[-2])  # the batch done is where the last block got to
    fractions = [fraction for fraction, _ in reports]
    assert fractions == sorted(fractions)


def test_no_hook_is_left_on_the_models_after_a_run_a_failure_or_an_interrupt(tmp_path):
    pipeline = ModelPipeline()
    before = hook_count(pipeline)
    make_tracked_run(pipeline)[0].upscale_images(one_image(), make_request(tmp_path), False, False)
    assert hook_count(pipeline) == before

    def fail():
        raise RuntimeError("CUDA out of memory")

    pipeline = ModelPipeline(on_block=fail)
    before = hook_count(pipeline)
    outcome = make_tracked_run(pipeline)[0].upscale_images(one_image(), make_request(tmp_path), False, False)
    assert outcome.failure == "Runtime Error: RuntimeError"
    assert hook_count(pipeline) == before

    pressed = []
    pipeline = ModelPipeline(on_block=lambda: pressed.append(1))
    before = hook_count(pipeline)
    run, _ = make_tracked_run(pipeline, is_interrupted=lambda: bool(pressed))
    assert run.upscale_images(one_image(), make_request(tmp_path), False, False).failure == INTERRUPTED_MESSAGE
    assert hook_count(pipeline) == before


def test_only_container_modules_are_watched(tmp_path):
    """Layers are not watched: a check before every layer slowed a real tiled decode several times over."""
    pipeline = ModelPipeline(blocks=2)
    seen = []
    run, _ = make_tracked_run(pipeline)
    original = run.module_called
    run.module_called = lambda module: (seen.append(type(module).__name__), original(module))
    run.upscale_images(one_image(), make_request(tmp_path), False, False)
    assert "Identity" not in seen
    assert seen.count("Block") == 2 and seen.count("Dit") == 1
    assert seen.count("Sequential") == 3 * 3  # each of three tiles: the decoder and its two stages


def test_interrupt_inside_the_decode_stops_before_the_next_tile(tmp_path):
    pipeline = ModelPipeline()
    pressed = []
    tiles = []

    def press_on_second_tile():
        tiles.append(1)
        if len(tiles) == 2:
            pressed.append(1)

    pipeline.on_decode = press_on_second_tile
    run, _ = make_tracked_run(pipeline, is_interrupted=lambda: bool(pressed))
    assert run.upscale_images(one_image(), make_request(tmp_path), False, False).failure == INTERRUPTED_MESSAGE
    assert len(tiles) == 2


class DroppingPipeline(ModelPipeline):
    """Drops its DiT after the DiT phase, as the vendored pipeline does without caching, and records in the decode
    phase whether anything still keeps it alive."""

    def upscale_all_batches(self, runner, ctx, progress_callback, **kwargs):
        ctx = super().upscale_all_batches(runner, ctx, progress_callback, **kwargs)
        self.dit_ref = weakref.ref(self.dit)
        self.dit = runner.dit = None
        return ctx

    def decode_all_batches(self, runner, ctx, progress_callback, **kwargs):
        gc.collect()
        self.dit_alive_in_decode = self.dit_ref() is not None
        return ctx


def test_a_dit_dropped_after_its_phase_is_freed_before_the_decode(tmp_path):
    """The hooks must not keep a dropped model (and its VRAM) alive through the later phases."""
    pipeline = DroppingPipeline()
    run, _ = make_tracked_run(pipeline)
    assert run.upscale_images(one_image(), make_request(tmp_path), False, False).failure is None
    assert pipeline.dit_alive_in_decode is False


def test_cached_models_are_not_stopped_after_the_run(tmp_path):
    pipeline = ModelPipeline()
    pressed = []
    run, _ = make_tracked_run(pipeline, is_interrupted=lambda: bool(pressed))
    run.upscale_images(one_image(), make_request(tmp_path), False, False)
    pressed.append(1)
    assert pipeline.dit(torch.zeros(1)).item() == 4
    pipeline.vae.decode(1)


@pytest.mark.parametrize("model", [None, SimpleNamespace(blocks=[]), "vae", 3])
def test_anything_that_is_not_a_module_has_no_containers(model):
    assert container_modules(model) == []


@pytest.mark.parametrize("runner", [None, SimpleNamespace(), SimpleNamespace(dit=None),
                                    SimpleNamespace(dit=SimpleNamespace(blocks=5))])
def test_a_runner_without_dit_blocks_gives_no_block_types(runner):
    assert dit_block_types(runner) == ((), 0)


def test_dit_block_types_are_read_through_the_wrapper():
    assert dit_block_types(SimpleNamespace(dit=SimpleNamespace(dit_model=Dit(3)))) == ((Block,), 3)


def test_attention_availability_reads_the_vendored_flags():
    available = attention_availability(SimpleNamespace(FLASH_ATTN_2_AVAILABLE=False, SAGE_ATTN_2_AVAILABLE=True))
    assert available == {AttentionMode.SDPA: True, AttentionMode.FLASH_ATTN_2: False,
                         AttentionMode.FLASH_ATTN_3: False, AttentionMode.SAGEATTN_2: True,
                         AttentionMode.SAGEATTN_3: False}

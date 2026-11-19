"""Running the pipeline over images — `seedvr2_neo/upscale_run.py`. The whole run through the Forge script is covered in
`tests/scripts/test_seedvr2_script.py`; here are the parts that script tests cannot reach directly."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import build_memory_plan
from seedvr2_neo.model_lifecycle import ModelLifecycle
from seedvr2_neo.upscale_run import INTERRUPTED_MESSAGE, HostHooks, UpscaleInterrupted, UpscaleRequest, UpscaleRun, \
    images_to_tensor, tensor_to_images
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


def test_the_progress_callback_raises_only_when_interrupted():
    make_run(Pipeline())[0].check_interruption(1, 2, "any", key="args")
    with pytest.raises(UpscaleInterrupted):
        make_run(Pipeline(), interrupted=True)[0].check_interruption()

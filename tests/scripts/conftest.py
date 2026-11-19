"""Fixtures that load `scripts/seedvr2_script.py` against the Forge, Gradio and pipeline stubs of `forge_stubs.py`.

Every test gets fresh stubs and a fresh import of the script, so its process-wide model lifecycle starts empty.
"""

import importlib.util
import os
import sys

import pytest

from forge_stubs import FakePipeline, Forge, ScriptRunner, StableDiffusionProcessingImg2Img, \
    StableDiffusionProcessingTxt2Img, forge_modules, pipeline_modules

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "seedvr2_script.py")
SCRIPT_MODULE_NAME = "seedvr2_script"

DIT = "seedvr2_ema_3b-Q4_K_M.gguf"
VAE = "ema_vae_fp16.safetensors"


@pytest.fixture
def forge(tmp_path, monkeypatch) -> Forge:
    """Forge's state with a model folder holding one DiT and one VAE file."""
    models_path = tmp_path / "models"
    model_folder = models_path / "SeedVR2"
    model_folder.mkdir(parents=True)
    for name in (DIT, VAE):
        (model_folder / name).write_bytes(b"not a real model")
    stubs = Forge(str(models_path))
    for name, module in forge_modules(stubs).items():
        monkeypatch.setitem(sys.modules, name, module)
    return stubs


@pytest.fixture
def pipeline(forge, monkeypatch) -> FakePipeline:
    fake = FakePipeline(forge)
    for name, module in pipeline_modules(fake).items():
        monkeypatch.setitem(sys.modules, name, module)
    return fake


@pytest.fixture
def script_module(forge, pipeline, monkeypatch):
    """The script freshly imported against the stubs."""
    spec = importlib.util.spec_from_file_location(SCRIPT_MODULE_NAME, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, SCRIPT_MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(script_module):
    return script_module.Script()


class OtherScript:
    """Another script in `p.scripts`, which must stay there while the base images are generated."""

    def title(self):
        return "Other script"


@pytest.fixture
def make_txt2img(script):
    def make(batch_size=1, n_iter=1, **fields):
        p = StableDiffusionProcessingTxt2Img(prompt="a cat", negative_prompt="blurry", batch_size=batch_size,
                                             n_iter=n_iter, **fields)
        p.scripts = ScriptRunner([OtherScript(), script])
        return p

    return make


def make_img2img(init_images, **fields):
    return StableDiffusionProcessingImg2Img(prompt="a dog", negative_prompt="", seed=-1, init_images=init_images,
                                            **fields)


def run_args(**overrides):
    """`Script.run` arguments after `p`, in the order `Script.ui` returns its components."""
    args = dict(dit_model_name=DIT, vae_model_name=VAE, seed=-1, resolution=1080, input_noise=0.0, latent_noise=0.0,
                force_reload=False, unload_sd=True, save_original=False, use_tile_vae=True, tile_size=1024,
                tile_overlap=128, debug_mode=False, after_run="Unload SeedVR2", blocks_to_swap=0,
                swap_io_components=False)
    args.update(overrides)
    return args

"""Stand-ins for Forge Neo, Gradio and the vendored SeedVR2 pipeline, so `scripts/seedvr2_script.py` runs in pytest.

Only the names the script uses exist here, with the shapes Forge Neo has; the facts and their sources are in
`docs/external-apis/forge-neo.md` ("Names `scripts/seedvr2_script.py` uses"). A stub that drifts from Forge makes tests
pass against an API that does not exist, so change a shape here only together with that document.

The pipeline is replaced at the module level (`seedvr2_vendor.core.*` in `sys.modules`), not inside the script, so the
tests do not depend on where the script imports it.
"""

import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image

REFRESH_SYMBOL = "\U0001f504"


# --- gradio ---------------------------------------------------------------------------------------------------------

class Component:
    """Any Gradio component or layout block. Forge Neo patches every component to accept `tooltip=` and drops unknown
    keyword arguments with a warning, so any keyword is accepted here too."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.events: List[Dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        self.events.append({"event": "click", "fn": fn, "inputs": inputs, "outputs": outputs})

    def change(self, fn=None, inputs=None, outputs=None, **kwargs):
        self.events.append({"event": "change", "fn": fn, "inputs": inputs, "outputs": outputs})


def make_gradio() -> types.ModuleType:
    gradio = types.ModuleType("gradio")
    for name in ("Row", "Accordion", "Dropdown", "Markdown", "Number", "Slider", "Checkbox", "Radio", "Button"):
        setattr(gradio, name, type(name, (Component,), {}))
    gradio.update = lambda **kwargs: {"__type__": "update", **kwargs}
    return gradio


# --- modules.shared, modules.devices, modules.sd_models -------------------------------------------------------------

class Options:
    """`shared.opts` with Forge's defaults for the options the script reads."""

    def __init__(self):
        self.samples_save = True
        self.samples_format = "png"
        self.save_to_dirs = True
        self.enable_pnginfo = True
        self.return_grid = True  # read by the stub `process_images` only


class State:
    """`shared.state` (`modules/shared_state.py`); the script reads only `interrupted`."""

    def __init__(self):
        self.skipped = False
        self.interrupted = False
        self.stopping_generation = False
        self.job = ""
        self.job_no = 0
        self.job_count = 0
        self.job_timestamp = "0"


@dataclass
class SavedImage:
    """One `images.save_image` call."""

    image: Image.Image
    path: str
    seed: Any
    prompt: Any
    extension: str
    info: Any
    suffix: str
    save_to_dirs: Optional[bool]


class Forge:
    """The mutable state of every stub module, reset for each test, plus what the script did with it."""

    def __init__(self, models_path: str):
        self.models_path = models_path
        self.opts = Options()
        self.state = State()
        self.saved: List[SavedImage] = []
        self.torch_gc_calls = 0
        self.sd_unloads = 0
        self.process_images_calls = 0
        self.scripts_during_generation: List[List[Any]] = []
        self.save_error: Optional[Exception] = None
        self.interrupt_generation = False  # the user presses Interrupt while Forge makes the base images

    def save_image(self, image, path, basename, seed=None, prompt=None, extension="png", info=None,
                   short_filename=False, no_prompt=False, grid=False, pnginfo_section_name="parameters", p=None,
                   existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
        """`modules.images.save_image`: same signature and return value `(fullfn, txt_fullfn)`; records the call
        instead of writing a file, and sets `already_saved_as` on the image as Forge does."""
        if self.save_error is not None:
            raise self.save_error
        fullfn = f"{path}/{len(self.saved):05d}-{seed}{suffix}.{extension}"
        self.saved.append(SavedImage(image, path, seed, prompt, extension, info, suffix, save_to_dirs))
        image.already_saved_as = fullfn
        return fullfn, None

    def torch_gc(self):
        self.torch_gc_calls += 1

    def unload_model_weights(self, *args, **kwargs):
        self.sd_unloads += 1


# --- modules.processing -----------------------------------------------------------------------------------------------

class Processed:
    """`modules.processing.Processed` with the constructor signature of Forge Neo. The real one reads ~25 fields of
    `p`; this one keeps only what the script and the tests read."""

    def __init__(self, p, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None,
                 all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments="",
                 extra_images_list=[]):
        self.images = images_list
        self.seed = seed
        self.info = info
        self.infotexts = infotexts or [info]
        self.index_of_first_image = index_of_first_image
        self.comments = "".join(f"{comment}\n" for comment in p.comments)
        self.all_seeds = all_seeds or p.all_seeds


class ScriptRunner:
    """`p.scripts`: Forge's `ScriptRunner`, whose `scripts` attribute is a plain list."""

    def __init__(self, scripts=()):
        self.scripts = list(scripts)


@dataclass
class StableDiffusionProcessing:
    """The `p` fields the script touches, with Forge's defaults (`modules/processing.py`)."""

    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    width: int = 8
    height: int = 8
    batch_size: int = 1
    n_iter: int = 1
    outpath_samples: Optional[str] = "outputs"
    do_not_save_samples: bool = False
    override_settings: Dict[str, Any] = field(default_factory=dict)
    comments: Dict[str, int] = field(default_factory=dict)
    scripts: Optional[ScriptRunner] = None
    all_seeds: Optional[List[int]] = None
    all_prompts: Optional[List[str]] = None

    def comment(self, text):
        self.comments[text] = 1


@dataclass
class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    pass


@dataclass
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    init_images: Optional[list] = None


def base_image_color(index: int) -> tuple:
    """A distinct colour per generated image, so a test can tell which result came from which base image."""
    return (10 + 40 * index, 200 - 30 * index, 100)


def make_process_images(forge: Forge) -> Callable:
    def process_images(p):
        """Forge's txt2img: `batch_size * n_iter` images with seeds from `p.seed`, and a grid in front of them when
        there is more than one and grids are returned (then `index_of_first_image` is 1)."""
        forge.process_images_calls += 1
        forge.scripts_during_generation.append(list(p.scripts.scripts) if p.scripts is not None else [])
        if forge.interrupt_generation:
            forge.state.interrupted = True
        count = p.batch_size * p.n_iter
        first_seed = 1000 if p.seed == -1 else p.seed
        p.all_seeds = [first_seed + index for index in range(count)]
        p.all_prompts = [p.prompt] * count
        images = [Image.new("RGB", (p.width, p.height), base_image_color(index)) for index in range(count)]
        infotexts = [f"{p.prompt}\nSteps: 20, Seed: {seed}, Size: {p.width}x{p.height}" for seed in p.all_seeds]
        index_of_first_image = 0
        if count > 1 and forge.opts.return_grid:
            images.insert(0, Image.new("RGB", (p.width * count, p.height), (0, 0, 0)))
            infotexts.insert(0, infotexts[0])
            index_of_first_image = 1
        return Processed(p, images, p.all_seeds[0], infotexts[0], index_of_first_image=index_of_first_image,
                         infotexts=infotexts)

    return process_images


# --- modules.scripts, modules.ui_common, modules.ui_components ---------------------------------------------------------

class Script:
    """`modules.scripts.Script`: the class attributes Forge sets, and `elem_id` built from the tab and the title."""

    name = None
    section = None
    filename = None
    args_from = None
    args_to = None
    alwayson = False
    is_img2img = False
    tabname = None
    group = None

    def elem_id(self, item_id):
        tab = ("img2img" if self.is_img2img else "txt2img") + "_" if self.show(True) == self.show(False) else ""
        title = "".join(char for char in self.title().lower().replace(" ", "_") if char.isalnum() or char == "_")
        return f"script_{tab}{title}_{item_id}"


def make_ui_components(gradio: types.ModuleType) -> types.ModuleType:
    module = types.ModuleType("modules.ui_components")

    class ToolButton(gradio.Button):
        def __init__(self, value="", *args, elem_classes=None, **kwargs):
            super().__init__(*args, elem_classes=["tool", *(elem_classes or [])], value=value, **kwargs)

    class FormRow(gradio.Row):
        pass

    module.ToolButton = ToolButton
    module.FormRow = FormRow
    return module


# --- the vendored pipeline ----------------------------------------------------------------------------------------------

UPSCALE_FACTOR = 2


class FakeModelCache:
    """The part of the vendored `GlobalModelCache` the script and `ModelLifecycle` use."""

    def __init__(self):
        self.cached = set()
        self.removed: List[str] = []

    def store(self, node_id):
        self.cached.add(node_id)

    def remove_dit(self, dit_config):
        return self._remove(dit_config["node_id"])

    def remove_vae(self, vae_config):
        return self._remove(vae_config["node_id"])

    def _remove(self, node_id):
        self.removed.append(node_id)
        if node_id in self.cached:
            self.cached.discard(node_id)
            return True
        return False


class FakePipeline:
    """The four phases of the vendored pipeline. Upscales by `UPSCALE_FACTOR` (pixel repeat), calls the progress
    callback in every phase, and can fail or be interrupted in a chosen phase (of every pass, or of one pass)."""

    PHASES = ("encode", "upscale", "decode", "postprocess")

    def __init__(self, forge: Forge):
        self.forge = forge
        self.cache = FakeModelCache()
        self.passes: List[torch.Tensor] = []  # the input tensor of every pass
        self.runner_kwargs: List[Dict[str, Any]] = []
        self.generation_kwargs: List[Dict[str, Any]] = []
        self.fail_in: Optional[str] = None
        self.failure: Exception = RuntimeError("CUDA out of memory at C:/Users/someone/models")
        self.interrupt_in: Optional[str] = None
        self.only_in_pass: Optional[int] = None  # 0-based pass the failure or interrupt happens in; None = every pass
        self.phases_run: List[str] = []

    def _phase(self, name, progress_callback):
        self.phases_run.append(name)
        in_chosen_pass = self.only_in_pass is None or self.only_in_pass == len(self.passes) - 1
        if self.interrupt_in == name and in_chosen_pass:
            self.forge.state.interrupted = True
        if self.fail_in == name and in_chosen_pass:
            raise self.failure
        if progress_callback is not None:
            progress_callback(1, 1, 1, name)

    def setup_generation_context(self, **kwargs):
        return {"setup": kwargs}

    def prepare_runner(self, **kwargs):
        self.runner_kwargs.append(kwargs)
        for node_id, cache_flag in ((kwargs["dit_id"], kwargs["dit_cache"]), (kwargs["vae_id"], kwargs["vae_cache"])):
            if cache_flag:
                self.cache.store(node_id)
        return object(), {"cache": True}

    def compute_generation_info(self, ctx, images, **kwargs):
        self.generation_kwargs.append(kwargs)
        self.passes.append(images)
        return images, {}

    def encode_all_batches(self, runner, ctx, images, progress_callback=None, **kwargs):
        self._phase("encode", progress_callback)
        return {**ctx, "frames": images}

    def upscale_all_batches(self, runner, ctx, progress_callback=None, **kwargs):
        self._phase("upscale", progress_callback)
        return ctx

    def decode_all_batches(self, runner, ctx, progress_callback=None, **kwargs):
        self._phase("decode", progress_callback)
        return ctx

    def postprocess_all_batches(self, ctx, progress_callback=None, **kwargs):
        self._phase("postprocess", progress_callback)
        frames = ctx["frames"].to(dtype=torch.float32)
        upscaled = frames.repeat_interleave(UPSCALE_FACTOR, dim=1).repeat_interleave(UPSCALE_FACTOR, dim=2)
        return {**ctx, "final_video": upscaled.to(dtype=torch.float16)}


class Debug:
    def __init__(self, enabled=False):
        self.enabled = enabled


def pipeline_modules(pipeline: FakePipeline) -> Dict[str, types.ModuleType]:
    """`sys.modules` entries for the vendored modules the script imports its pipeline from."""
    debug = types.ModuleType("seedvr2_vendor.utils.debug")
    debug.Debug = Debug
    utils = types.ModuleType("seedvr2_vendor.core.generation_utils")
    utils.setup_generation_context = pipeline.setup_generation_context
    utils.prepare_runner = pipeline.prepare_runner
    utils.compute_generation_info = pipeline.compute_generation_info
    phases = types.ModuleType("seedvr2_vendor.core.generation_phases")
    phases.encode_all_batches = pipeline.encode_all_batches
    phases.upscale_all_batches = pipeline.upscale_all_batches
    phases.decode_all_batches = pipeline.decode_all_batches
    phases.postprocess_all_batches = pipeline.postprocess_all_batches
    cache = types.ModuleType("seedvr2_vendor.core.model_cache")
    cache.get_global_cache = lambda: pipeline.cache
    return {module.__name__: module for module in (debug, utils, phases, cache)}


# --- installing the stubs -------------------------------------------------------------------------------------------------

def forge_modules(forge: Forge) -> Dict[str, types.ModuleType]:
    """`sys.modules` entries for `gradio` and every `modules.*` name the script imports, bound to `forge`."""
    gradio = make_gradio()
    modules = types.ModuleType("modules")
    modules.__path__ = []  # a package, so `import modules.sd_models` resolves through sys.modules

    shared = types.ModuleType("modules.shared")
    shared.models_path = forge.models_path
    shared.opts = forge.opts
    shared.state = forge.state

    devices = types.ModuleType("modules.devices")
    devices.device = torch.device("cpu")
    devices.torch_gc = forge.torch_gc

    sd_models = types.ModuleType("modules.sd_models")
    sd_models.unload_model_weights = forge.unload_model_weights

    images = types.ModuleType("modules.images")
    images.save_image = forge.save_image

    processing = types.ModuleType("modules.processing")
    processing.Processed = Processed
    processing.StableDiffusionProcessing = StableDiffusionProcessing
    processing.StableDiffusionProcessingTxt2Img = StableDiffusionProcessingTxt2Img
    processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img
    processing.process_images = make_process_images(forge)

    scripts = types.ModuleType("modules.scripts")
    scripts.Script = Script
    scripts.ScriptRunner = ScriptRunner

    ui_common = types.ModuleType("modules.ui_common")
    ui_common.refresh_symbol = REFRESH_SYMBOL

    ui_components = make_ui_components(gradio)

    submodules = {"shared": shared, "devices": devices, "sd_models": sd_models, "images": images,
                  "processing": processing, "scripts": scripts, "ui_common": ui_common, "ui_components": ui_components}
    for name, module in submodules.items():
        setattr(modules, name, module)
    return {"gradio": gradio, "modules": modules, **{f"modules.{name}": module for name, module in submodules.items()}}

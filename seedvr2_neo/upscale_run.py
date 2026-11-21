"""Running the vendored SeedVR2 pipeline over a list of images: passes, the four phases, errors and interrupts.

The only module here that imports torch: it turns PIL images into the pipeline's tensors and back. What it needs from
Forge — the device, the memory release, the interrupt flag and the progress bar — the script (`scripts/seedvr2_script.py`) passes in as
`HostHooks`, so this runs in tests with a fake pipeline. Which images go in: `seedvr2_neo/run_inputs.py`; when the
cached models are dropped: `seedvr2_neo/model_lifecycle.py`.
"""

import os
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from seedvr2_neo.memory_plan import CPU_DEVICE, MemoryPlan
from seedvr2_neo.model_lifecycle import DIT_CACHE_ID, VAE_CACHE_ID, ModelLifecycle
from seedvr2_neo.pipeline_options import AttentionMode, PipelineOptions
from seedvr2_neo.run_inputs import group_by_shape, split_into_passes
from seedvr2_neo.run_progress import DECODE_PHASE, ENCODE_PHASE, POSTPROCESS_PHASE, UPSCALE_PHASE, ProgressReport, \
    RunProgress
from seedvr2_neo.upscale_settings import OutputTarget, UpscaleSettings

# Why images come back without upscaling when the user pressed Interrupt (infotext and result comment).
INTERRUPTED_MESSAGE = "Interrupted"


def import_pipeline() -> SimpleNamespace:
    """Import the vendored pipeline. The script calls it right after the model check in run(), so a broken install
    fails before any generation or SD unload; not at module level, so the UI still loads and shows the error instead
    of the script vanishing."""
    from seedvr2_vendor.utils.debug import Debug
    from seedvr2_vendor.core.generation_utils import setup_generation_context, prepare_runner, compute_generation_info
    from seedvr2_vendor.core.generation_phases import encode_all_batches, upscale_all_batches, decode_all_batches, postprocess_all_batches
    from seedvr2_vendor.core.model_cache import get_global_cache
    from seedvr2_vendor.optimization import compatibility
    return SimpleNamespace(
        Debug=Debug, setup_generation_context=setup_generation_context, prepare_runner=prepare_runner,
        compute_generation_info=compute_generation_info, encode_all_batches=encode_all_batches,
        upscale_all_batches=upscale_all_batches, decode_all_batches=decode_all_batches,
        postprocess_all_batches=postprocess_all_batches, get_global_cache=get_global_cache,
        attention_available=attention_availability(compatibility),
    )


# The vendored `optimization/compatibility.py` flag that says a backend imported.
ATTENTION_FLAGS = {AttentionMode.FLASH_ATTN_2: "FLASH_ATTN_2_AVAILABLE", AttentionMode.FLASH_ATTN_3: "FLASH_ATTN_3_AVAILABLE",
                   AttentionMode.SAGEATTN_2: "SAGE_ATTN_2_AVAILABLE", AttentionMode.SAGEATTN_3: "SAGE_ATTN_3_AVAILABLE"}


def attention_availability(compatibility) -> Dict[AttentionMode, bool]:
    """Which attention backends the vendored pipeline imported successfully (its module-level flags)."""
    return {AttentionMode.SDPA: True,
            **{mode: bool(getattr(compatibility, flag, False)) for mode, flag in ATTENTION_FLAGS.items()}}


@dataclass(frozen=True)
class UpscaleRequest:
    """Everything one pipeline run needs from the UI, already converted to plain types."""

    dit_model: str
    vae_model: str
    model_dir: str
    memory_plan: MemoryPlan
    seed: int
    settings: UpscaleSettings
    options: PipelineOptions = PipelineOptions()


@dataclass(frozen=True)
class HostHooks:
    """What the run takes from the host WebUI."""

    device: torch.device  # the DiT and VAE run on it
    release_memory: Callable[[], None]  # collect dropped objects and return the allocator's cached VRAM
    is_interrupted: Callable[[], bool]  # the user pressed Interrupt
    report_progress: ProgressReport = field(default=lambda fraction, text: None)  # the share of the run done, 0…1


@dataclass(frozen=True)
class UpscaleOutcome:
    """The upscaled images by input index and, when the run stopped early, why. Images of the passes that finished
    are kept even then."""

    upscaled: Dict[int, Image.Image]
    failure: Optional[str]


class UpscaleInterrupted(Exception):
    """Raised inside the pipeline to stop the run when the user presses Interrupt."""


@contextmanager
def watching_modules(modules: Iterable[torch.nn.Module], on_call: Callable[[torch.nn.Module], None]) -> Iterator[None]:
    """Call `on_call(module)` before the forward of each of `modules` while the block runs. The pipeline checks
    Interrupt only between images, and one image can spend minutes in the DiT or the tiled VAE; a check before each
    of these calls stops it within a fraction of a second. Always removed, so cached models keep no hook. Only the
    hook handles are kept (they hold the module's hook table weakly): a reference to the modules here would keep a
    model the pipeline drops after its phase alive, with whatever it still holds, until the last phase ends."""
    def hook(module, args) -> None:
        on_call(module)  # nothing is returned: a pre-hook's return value would replace the module's inputs

    handles = []
    try:
        for module in modules:
            handles.append(module.register_forward_pre_hook(hook))
        module = modules = None
        yield
    finally:
        for handle in handles:
            handle.remove()


def container_modules(model) -> List[torch.nn.Module]:
    """The modules of `model` that hold other modules (blocks, stages), not the layers inside them: a check before
    each runs a few hundred times per VAE tile or DiT pass, while one before every layer (or torch's process-wide
    hook) runs millions of times and slowed a tiled VAE decode several times over. The vendored loaders keep these
    objects when they load the weights (`load_state_dict(..., assign=True)`; GGUF swaps only the Linear layers), so
    hooks set before a phase stay on the model the phase runs. An empty list for anything that is not a module."""
    if not isinstance(model, torch.nn.Module):
        return []
    return [module for module in model.modules() if next(module.children(), None) is not None]


def runner_dit(runner: object) -> Optional[object]:
    """The runner's DiT, unwrapped from the vendored fp8 / GGUF wrapper (`dit_model`), or None."""
    model = getattr(runner, "dit", None)
    return getattr(model, "dit_model", model)


def dit_block_types(runner: object) -> Tuple[Tuple[type, ...], int]:
    """The classes of the DiT's transformer blocks and how many blocks one forward runs, or ((), 0) when the runner
    holds no DiT with blocks."""
    blocks = getattr(runner_dit(runner), "blocks", None)
    try:
        return tuple({type(block) for block in blocks}), len(blocks)
    except TypeError:
        return (), 0


def images_to_tensor(group_images: List[Image.Image]) -> torch.Tensor:
    """Same-sized PIL images -> one [N, H, W, C] float16 tensor in [0, 1], the pipeline's input format. Each image is
    made float16 before stacking, so no float32 copy of the whole pass exists."""
    to_tensor = transforms.ToTensor()
    frames = [to_tensor(image).to(dtype=torch.float16) for image in group_images]
    return torch.stack(frames).permute(0, 2, 3, 1)


def tensor_to_images(frames: torch.Tensor) -> List[Image.Image]:
    """The pipeline's [N, H, W, C] output -> N PIL images (RGB, or RGBA for 4 channels), one frame at a time."""
    frames = frames.detach().cpu()
    return [Image.fromarray((torch.clamp(frame.to(dtype=torch.float32), 0, 1).numpy() * 255.0).astype(np.uint8))
            for frame in frames]


def describe_target(size: Tuple[int, int], target: OutputTarget, count: int) -> str:
    """The console line naming the output size of one group of same-sized inputs."""
    width, height = size
    out_width, out_height = target.output_size(size)
    images = "image" if count == 1 else "images"
    return f"[SeedVR2] {count} {images} {width}x{height} -> {out_width}x{out_height}{target.raise_note()}"


class UpscaleRun:
    """One run of the pipeline over the inputs of one Generate press."""

    def __init__(self, pipeline: SimpleNamespace, lifecycle: ModelLifecycle, hooks: HostHooks) -> None:
        self._pipeline = pipeline
        self._lifecycle = lifecycle
        self._hooks = hooks
        self._progress = RunProgress([1], hooks.report_progress)
        self._dit_blocks: Tuple[type, ...] = ()  # set only while the DiT phase runs
        self._trim_module_ids: frozenset = frozenset()  # PROTOTYPE: ids of decoder stages while decoding

    def check_interruption(self) -> None:
        """Stop the run when the user pressed Interrupt. Only Interrupt does: Forge's "Stop generating"
        (`stopping_generation`) means "finish the current image", which should not abort the upscale."""
        if self._hooks.is_interrupted():
            raise UpscaleInterrupted("User interrupted generation.")

    def phase_progress(self, current: object = None, total: object = None, *args, **kwargs) -> None:
        """The pipeline's progress callback: `current` of `total` batches of the phase are done."""
        self.check_interruption()
        self._progress.batch_done(current, total)

    def module_called(self, module: torch.nn.Module) -> None:
        """Before a watched module's forward during the phases: Interrupt, and a DiT block moves the bar."""
        self.check_interruption()
        if self._dit_blocks and isinstance(module, self._dit_blocks):
            self._progress.block_started()
        if id(module) in self._trim_module_ids:
            torch.cuda.empty_cache()  # PROTOTYPE

    def upscale_images(self, inputs: List[Image.Image], request: UpscaleRequest, force_reload: bool,
                       debug_mode: bool) -> UpscaleOutcome:
        """Upscale the inputs in pipeline passes (same size and mode, within the pass budget) and unload the models
        after the last pass. With "Unload" each pass loads the models again; "Keep in RAM" reuses them."""
        model_cache = self._pipeline.get_global_cache()
        if self._lifecycle.prepare_run(model_cache, os.path.join(request.model_dir, request.dit_model),
                                       os.path.join(request.model_dir, request.vae_model), force_reload):
            print("[SeedVR2] Cached models dropped (Force Reload, changed model files or a UI reload); loading from disk.")

        upscaled: Dict[int, Image.Image] = {}
        failure = None
        try:
            debug = self._pipeline.Debug(enabled=debug_mode)
            groups = group_by_shape(inputs)
            targets = {size: request.settings.target_for(size) for size in {image.size for image in inputs}}
            for group in groups:
                size = inputs[group[0]].size
                print(describe_target(size, targets[size], len(group)))
            passes = split_into_passes(groups, inputs, lambda size: targets[size].output_pixels(size))
            if len(passes) > 1: print(f"[SeedVR2] {len(inputs)} images in {len(passes)} pipeline passes.")
            self._progress = RunProgress([len(pass_indices) for pass_indices in passes], self._hooks.report_progress)
            for pass_number, pass_indices in enumerate(passes):
                self.check_interruption()
                self._progress.start_pass(pass_number)
                pass_images = [inputs[index] for index in pass_indices]
                outputs = self._upscale_pass(pass_images, targets[pass_images[0].size], request, debug, debug_mode)
                upscaled.update(zip(pass_indices, outputs))
            if not debug_mode: print("[SeedVR2] Done.")
        except UpscaleInterrupted:
            print("\n⏹️ [SeedVR2] Process Interrupted by User.")
            failure = INTERRUPTED_MESSAGE
        except Exception as e:  # the run's boundary: the user gets their images back with the error in the infotext
            print(f"❌ [SeedVR2 Runtime Error]: {e}")
            traceback.print_exc()
            # Only the type goes into the infotext: the message may hold local paths or GPU details, and the infotext
            # leaves the machine with the image (API, shared PNG). The full text is in the console above.
            failure = f"Runtime Error: {type(e).__name__}"
        finally:
            kept = self._lifecycle.finish_run(model_cache, request.memory_plan.cache_models, failure is None)
            print("[SeedVR2] Models kept in RAM for the next run." if kept
                  else "[SeedVR2] Models unloaded, memory released.")
        return UpscaleOutcome(upscaled, failure)

    def _upscale_pass(self, pass_images: List[Image.Image], target: OutputTarget, request: UpscaleRequest, debug,
                      debug_mode: bool) -> List[Image.Image]:
        """Upscale images of one size and mode to `target` in one pipeline pass, as frames of one tensor."""
        frames = images_to_tensor(pass_images)
        output = self._upscale_tensor(frames, target, request, debug, debug_mode)
        outputs = tensor_to_images(output)
        if len(outputs) != len(pass_images):
            raise RuntimeError(f"the pipeline returned {len(outputs)} images for {len(pass_images)} inputs")
        return outputs

    def _upscale_tensor(self, img_tensor: torch.Tensor, target: OutputTarget, request: UpscaleRequest, debug,
                        debug_mode: bool) -> torch.Tensor:
        """Run the four SeedVR2 phases on a [N, H, W, C] tensor of N same-sized images and return the pipeline's output.

        With batch size 1 and no temporal overlap the pipeline treats every frame on its own, so the images do not
        change each other's content; the models are loaded once per pass.

        After each phase the dropped objects are collected and the allocator's cached VRAM is returned, so the next
        phase has the most free VRAM and is less likely to spill into shared memory."""
        pipeline, settings, plan = self._pipeline, request.settings, request.memory_plan
        runner, ctx = self._prepare_runner(request, debug)
        frames_tensor, _generation_info = pipeline.compute_generation_info(
            ctx=ctx, images=img_tensor, resolution=target.resolution, max_resolution=target.max_resolution,
            batch_size=1, uniform_batch_size=False, seed=request.seed,
            prepend_frames=0, temporal_overlap=0, debug=debug
        )

        color_correction = request.options.color_correction.value
        tiled_note = '(Tiled)' if settings.use_tile_vae else ''
        with watching_modules(container_modules(getattr(runner, "vae", None)) + container_modules(runner_dit(runner)),
                              self.module_called):
            if not debug_mode: print(f"[SeedVR2] Phase 1/4: Encoding {tiled_note}...")
            self._progress.start_phase(ENCODE_PHASE)
            ctx = pipeline.encode_all_batches(runner, ctx=ctx, images=frames_tensor, debug=debug, batch_size=1,
                                              uniform_batch_size=False, seed=request.seed,
                                              progress_callback=self.phase_progress, temporal_overlap=0,
                                              resolution=target.resolution, max_resolution=target.max_resolution,
                                              input_noise_scale=settings.input_noise, color_correction=color_correction)
            self._hooks.release_memory()

            if not debug_mode: print("[SeedVR2] Phase 2/4: Upscaling...")
            try:
                self._dit_blocks, blocks_per_image = dit_block_types(runner)
                self._progress.start_phase(UPSCALE_PHASE, blocks_per_image)
                ctx = pipeline.upscale_all_batches(runner, ctx=ctx, debug=debug, progress_callback=self.phase_progress,
                                                   seed=request.seed, latent_noise_scale=settings.latent_noise,
                                                   cache_model=plan.cache_models)
            finally:
                self._dit_blocks = ()
            self._hooks.release_memory()

            if not debug_mode: print(f"[SeedVR2] Phase 3/4: Decoding {tiled_note}...")
            self._progress.start_phase(DECODE_PHASE)
            decoder = getattr(getattr(runner, "vae", None), "decoder", None)
            stages = [decoder]
            if decoder is not None:
                stages.append(decoder.mid_block)
                for up_block in decoder.up_blocks:
                    stages.append(up_block)
                    stages.extend(up_block.children())
                    for child in up_block.children():
                        if isinstance(child, torch.nn.ModuleList):
                            stages.extend(child)
            self._trim_module_ids = frozenset(id(m) for m in stages if m is not None)
            print(f"[SeedVR2] PROTOTYPE trimming before {len(self._trim_module_ids)} decoder stages")
            decoder = stages = up_block = child = None
            try:
                ctx = pipeline.decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=self.phase_progress,
                                                  cache_model=plan.cache_models)
            finally:
                self._trim_module_ids = frozenset()
            self._hooks.release_memory()

            if not debug_mode: print("[SeedVR2] Phase 4/4: Post-processing...")
            self._progress.start_phase(POSTPROCESS_PHASE)
            ctx = pipeline.postprocess_all_batches(ctx=ctx, debug=debug, progress_callback=self.phase_progress,
                                                   color_correction=color_correction, prepend_frames=0,
                                                   temporal_overlap=0, batch_size=1)
        return ctx['final_video']

    def _prepare_runner(self, request: UpscaleRequest, debug: object) -> Tuple[object, Dict]:
        """A fresh generation context on the host's device, so nothing from a failed or interrupted run is reused, and
        the runner with the models (from the cache or the folder), tiling, BlockSwap and attention."""
        pipeline, settings, plan = self._pipeline, request.settings, request.memory_plan
        ctx = pipeline.setup_generation_context(
            dit_device=self._hooks.device,
            vae_device=self._hooks.device,
            dit_offload_device=plan.dit_offload_device,
            vae_offload_device=plan.vae_offload_device,
            tensor_offload_device=CPU_DEVICE,
            debug=debug
        )

        runner, cache_context = pipeline.prepare_runner(
            dit_model=request.dit_model,
            vae_model=request.vae_model,
            model_dir=request.model_dir,
            debug=debug,
            ctx=ctx,
            dit_cache=plan.cache_models,
            vae_cache=plan.cache_models,
            dit_id=DIT_CACHE_ID,
            vae_id=VAE_CACHE_ID,
            block_swap_config=plan.block_swap_config(),
            encode_tiled=settings.use_tile_vae,
            encode_tile_size=(settings.tile_size, settings.tile_size),
            encode_tile_overlap=(settings.tile_overlap, settings.tile_overlap),
            decode_tiled=settings.use_tile_vae,
            decode_tile_size=(settings.tile_size, settings.tile_size),
            decode_tile_overlap=(settings.tile_overlap, settings.tile_overlap),
            tile_debug="false", attention_mode=request.options.attention_mode.value,
            torch_compile_args_dit=None, torch_compile_args_vae=None
        )
        ctx['cache_context'] = cache_context
        # The pipeline's own Interrupt check at the top of each batch (upstream sets it only inside ComfyUI).
        ctx['interrupt_fn'] = self.check_interruption
        return runner, ctx

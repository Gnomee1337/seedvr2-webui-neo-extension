"""Running the vendored SeedVR2 pipeline over a list of images: passes, the four phases, errors and interrupts.

The only module here that imports torch: it turns PIL images into the pipeline's tensors and back. What it needs from
Forge — the device, the memory release and the interrupt flag — the script (`scripts/seedvr2_script.py`) passes in as
`HostHooks`, so this runs in tests with a fake pipeline. Which images go in: `seedvr2_neo/run_inputs.py`; when the
cached models are dropped: `seedvr2_neo/model_lifecycle.py`.
"""

import os
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from seedvr2_neo.memory_plan import CPU_DEVICE, MemoryPlan
from seedvr2_neo.model_lifecycle import DIT_CACHE_ID, VAE_CACHE_ID, ModelLifecycle
from seedvr2_neo.run_inputs import group_by_shape, split_into_passes
from seedvr2_neo.upscale_settings import OutputTarget, UpscaleSettings

# Why images come back without upscaling when the user pressed Interrupt (infotext and result comment).
INTERRUPTED_MESSAGE = "Interrupted"

# Fixed pipeline settings: the simplest configuration that works without optional packages (`sdpa` is PyTorch's own
# attention; LAB color correction keeps the input's colors).
COLOR_CORRECTION = "lab"
ATTENTION_MODE = "sdpa"


def import_pipeline() -> SimpleNamespace:
    """Import the vendored pipeline. The script calls it right after the model check in run(), so a broken install
    fails before any generation or SD unload; not at module level, so the UI still loads and shows the error instead
    of the script vanishing."""
    from seedvr2_vendor.utils.debug import Debug
    from seedvr2_vendor.core.generation_utils import setup_generation_context, prepare_runner, compute_generation_info
    from seedvr2_vendor.core.generation_phases import encode_all_batches, upscale_all_batches, decode_all_batches, postprocess_all_batches
    from seedvr2_vendor.core.model_cache import get_global_cache
    return SimpleNamespace(
        Debug=Debug, setup_generation_context=setup_generation_context, prepare_runner=prepare_runner,
        compute_generation_info=compute_generation_info, encode_all_batches=encode_all_batches,
        upscale_all_batches=upscale_all_batches, decode_all_batches=decode_all_batches,
        postprocess_all_batches=postprocess_all_batches, get_global_cache=get_global_cache,
    )


@dataclass(frozen=True)
class UpscaleRequest:
    """Everything one pipeline run needs from the UI, already converted to plain types."""

    dit_model: str
    vae_model: str
    model_dir: str
    memory_plan: MemoryPlan
    seed: int
    settings: UpscaleSettings


@dataclass(frozen=True)
class HostHooks:
    """What the run takes from the host WebUI."""

    device: torch.device  # the DiT and VAE run on it
    release_memory: Callable[[], None]  # collect dropped objects and return the allocator's cached VRAM
    is_interrupted: Callable[[], bool]  # the user pressed Interrupt


@dataclass(frozen=True)
class UpscaleOutcome:
    """The upscaled images by input index and, when the run stopped early, why. Images of the passes that finished
    are kept even then."""

    upscaled: Dict[int, Image.Image]
    failure: Optional[str]


class UpscaleInterrupted(Exception):
    """Raised from the pipeline's progress callback to stop the run when the user presses Interrupt."""


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
    raised = f" (raised to the {target.resolution} px minimum)" if target.raised_to_minimum else ""
    return f"[SeedVR2] {count} {images} {width}x{height} -> {out_width}x{out_height}{raised}"


class UpscaleRun:
    """One run of the pipeline over the inputs of one Generate press."""

    def __init__(self, pipeline: SimpleNamespace, lifecycle: ModelLifecycle, hooks: HostHooks) -> None:
        self._pipeline = pipeline
        self._lifecycle = lifecycle
        self._hooks = hooks

    def check_interruption(self, *args, **kwargs) -> None:
        """Passed to the pipeline as its progress callback. Only Interrupt stops the run: Forge's "Stop generating"
        (`stopping_generation`) means "finish the current image", which should not abort the upscale."""
        if self._hooks.is_interrupted():
            raise UpscaleInterrupted("User interrupted generation.")

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
            for pass_indices in passes:
                self.check_interruption()
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

        The generation context is built fresh every run, so nothing from a failed or interrupted run is reused. After
        each phase the dropped objects are collected and the allocator's cached VRAM is returned, so the next phase
        has the most free VRAM and is less likely to spill into shared memory."""
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
            tile_debug="false", attention_mode=ATTENTION_MODE,
            torch_compile_args_dit=None, torch_compile_args_vae=None
        )
        ctx['cache_context'] = cache_context

        frames_tensor, _generation_info = pipeline.compute_generation_info(
            ctx=ctx, images=img_tensor, resolution=target.resolution, max_resolution=target.max_resolution,
            batch_size=1, uniform_batch_size=False, seed=request.seed,
            prepend_frames=0, temporal_overlap=0, debug=debug
        )

        tiled_note = '(Tiled)' if settings.use_tile_vae else ''
        if not debug_mode: print(f"[SeedVR2] Phase 1/4: Encoding {tiled_note}...")
        ctx = pipeline.encode_all_batches(runner, ctx=ctx, images=frames_tensor, debug=debug, batch_size=1,
                                          uniform_batch_size=False, seed=request.seed,
                                          progress_callback=self.check_interruption, temporal_overlap=0,
                                          resolution=target.resolution, max_resolution=target.max_resolution,
                                          input_noise_scale=settings.input_noise, color_correction=COLOR_CORRECTION)
        self._hooks.release_memory()

        if not debug_mode: print("[SeedVR2] Phase 2/4: Upscaling...")
        ctx = pipeline.upscale_all_batches(runner, ctx=ctx, debug=debug, progress_callback=self.check_interruption,
                                           seed=request.seed, latent_noise_scale=settings.latent_noise,
                                           cache_model=plan.cache_models)
        self._hooks.release_memory()

        if not debug_mode: print(f"[SeedVR2] Phase 3/4: Decoding {tiled_note}...")
        ctx = pipeline.decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=self.check_interruption,
                                          cache_model=plan.cache_models)
        self._hooks.release_memory()

        if not debug_mode: print("[SeedVR2] Phase 4/4: Post-processing...")
        ctx = pipeline.postprocess_all_batches(ctx=ctx, debug=debug, progress_callback=self.check_interruption,
                                               color_correction=COLOR_CORRECTION, prepend_frames=0, temporal_overlap=0,
                                               batch_size=1)
        return ctx['final_video']

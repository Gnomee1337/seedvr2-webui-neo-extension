import gc
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import gradio as gr
from modules import scripts, processing, images, shared, devices
from modules.processing import Processed, create_infotext, StableDiffusionProcessingTxt2Img
from modules.ui_common import refresh_symbol
from modules.ui_components import FormRow, ToolButton
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import random

# Forge model management (used to unload the SD checkpoint and free VRAM)
import modules.sd_models

# Extension root directory. Forge already puts every extension folder on sys.path, which is what
# makes `seedvr2_neo` and the vendored `seedvr2_vendor` importable; the insert only covers a host that does not.
EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import (MAX_BLOCKS_TO_SWAP, AfterRun, MemoryPlan, build_memory_plan, is_7b_checkpoint,
                                     low_vram_warning)
from seedvr2_neo.model_discovery import ModelScan, describe_missing_models, find_selection_error, scan_models
from seedvr2_neo.model_lifecycle import DIT_CACHE_ID, VAE_CACHE_ID, ModelLifecycle


def scan_seedvr2_models() -> ModelScan:
    """Rescan on every call: the user may add files or create the folder while the WebUI runs."""
    return scan_models(shared.models_path, EXTENSION_ROOT)


def empty_result(p, message: str) -> Processed:
    """A result with no images. `infotexts` is filled because the txt2img ✨ (hires upscale) button reads infotexts[0]."""
    return Processed(p, [], p.seed, message, infotexts=[message])


def release_freed_memory() -> None:
    """Return memory of dropped models and tensors: Python objects first, then the allocator's cached VRAM."""
    gc.collect()
    devices.torch_gc()


# Module-level: Forge creates one Script per tab, while the vendored model cache is shared by the whole process.
MODEL_LIFECYCLE = ModelLifecycle(LoadedModelTracker(), release_freed_memory)


def import_pipeline() -> SimpleNamespace:
    """Import the vendored pipeline. Done first in run(), so a broken install fails before any generation or unload;
    not at module level, so the UI still loads and shows the error instead of the script vanishing."""
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


def get_total_vram_bytes() -> Optional[int]:
    """Total memory of Forge's GPU, or None on a non-CUDA device or when it cannot be read."""
    if devices.device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(devices.device).total_memory)
    except (RuntimeError, AssertionError) as e:
        print(f"[SeedVR2] Cannot read the GPU memory size: {e}")
        return None


def describe_vram_warning(scan: ModelScan, dit_model_name: Optional[str]) -> str:
    """The 7B-on-a-small-GPU note for the selected DiT, or an empty string."""
    if not dit_model_name or scan.folder is None or dit_model_name not in scan.dit_models:
        return ""
    is_7b = is_7b_checkpoint(dit_model_name, os.path.join(scan.folder, dit_model_name))
    return low_vram_warning(is_7b, get_total_vram_bytes()) or ""


@dataclass(frozen=True)
class UpscaleRequest:
    """Everything one pipeline run needs from the UI, already converted to plain types."""

    dit_model: str
    vae_model: str
    model_dir: str
    memory_plan: MemoryPlan
    seed: int
    resolution: int
    input_noise: float
    latent_noise: float
    use_tile_vae: bool
    tile_size: int
    tile_overlap: int


# Custom exception used to stop SeedVR2 cleanly when the user presses Interrupt
class SeedVR2Interrupted(Exception):
    pass

class Script(scripts.Script):
    def title(self):
        return "SeedVR2 Native Upscaler (24G)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        scan = scan_seedvr2_models()
        missing_note = describe_missing_models(scan)
        vram_warning = describe_vram_warning(scan, scan.default_dit)

        with gr.Row():
            dit_model = gr.Dropdown(label="DiT Model", choices=list(scan.dit_models), value=scan.default_dit)
            vae_model = gr.Dropdown(label="VAE Model", choices=list(scan.vae_models), value=scan.default_vae)
            refresh_models = ToolButton(value=refresh_symbol, elem_id=self.elem_id("refresh_models"),
                                        tooltip="Rescan the SeedVR2 model folder")
        missing_models_note = gr.Markdown(missing_note, visible=bool(missing_note))
        vram_warning_note = gr.Markdown(vram_warning, visible=bool(vram_warning))
        refresh_models.click(fn=self._refresh_models, inputs=[dit_model, vae_model],
                             outputs=[dit_model, vae_model, missing_models_note])
        dit_model.change(fn=self._update_vram_warning, inputs=[dit_model], outputs=[vram_warning_note])

        with gr.Row():
            seed = gr.Number(label="Seed (-1 = Random/Follow)", value=-1, precision=0)
            resolution = gr.Slider(label="Upscale Resolution (Shortest Edge)", minimum=512, maximum=3840, step=64, value=1080)

        with gr.Row():
            unload_sd = gr.Checkbox(label="Unload SD Checkpoint", value=True, elem_classes="force-unload",
                                    info="Frees VRAM; the checkpoint reloads on the next generation")
            save_original = gr.Checkbox(label="Save Extra Copy of Original", value=False)
            force_reload = gr.Checkbox(label="Force Reload", value=False)
            debug_mode = gr.Checkbox(label="Show Debug Logs", value=False)

        with gr.Row():
            after_run = gr.Radio(label="After run", choices=[option.value for option in AfterRun],
                                 value=AfterRun.UNLOAD.value,
                                 info="Unload frees VRAM and RAM; Keep in RAM makes the next run start faster")
            blocks_to_swap = gr.Slider(label="BlockSwap: DiT blocks kept in RAM", minimum=0,
                                       maximum=MAX_BLOCKS_TO_SWAP, step=1, value=0)
            swap_io_components = gr.Checkbox(label="BlockSwap: I/O layers too", value=False)

        with gr.Accordion("Advanced Settings (Noise & Tiling)", open=False):
            with gr.Row():
                input_noise = gr.Slider(label="Input Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                latent_noise = gr.Slider(label="Latent Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0)

            gr.Markdown("ℹ️ **Tiling reduces VRAM usage for huge resolutions.**")
            with gr.Row():
                use_tile_vae = gr.Checkbox(label="Enable VAE Tiling", value=True)
                tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=2048, step=128, value=1024)
                tile_overlap = gr.Slider(label="Tile Overlap", minimum=64, maximum=512, step=32, value=128)

        return [dit_model, vae_model, seed, resolution, input_noise, latent_noise, force_reload, unload_sd, save_original,
                use_tile_vae, tile_size, tile_overlap, debug_mode, after_run, blocks_to_swap, swap_io_components]

    @staticmethod
    def _refresh_models(current_dit, current_vae):
        """Rescan the folder; keep the current choices if their files are still there."""
        scan = scan_seedvr2_models()
        note = describe_missing_models(scan)
        return (
            gr.update(choices=list(scan.dit_models), value=scan.keep_or_default_dit(current_dit)),
            gr.update(choices=list(scan.vae_models), value=scan.keep_or_default_vae(current_vae)),
            gr.update(value=note, visible=bool(note)),
        )

    @staticmethod
    def _update_vram_warning(dit_model_name):
        warning = describe_vram_warning(scan_seedvr2_models(), dit_model_name)
        return gr.update(value=warning, visible=bool(warning))

    def run(self, p, dit_model_name, vae_model_name, seed, resolution, input_noise, latent_noise, force_reload, unload_sd,
            save_original, use_tile_vae, tile_size, tile_overlap, debug_mode, after_run=AfterRun.UNLOAD.value,
            blocks_to_swap=0, swap_io_components=False):
        # Checked before the base image is generated, so a bad selection costs no generation time
        scan = scan_seedvr2_models()
        setup_error = find_selection_error(scan, dit_model_name, vae_model_name)
        if setup_error:
            print(f"❌ [SeedVR2] {setup_error}")
            return empty_result(p, f"SeedVR2: {setup_error}")
        print(f"[SeedVR2] Model folder: {scan.folder} | DiT: {dit_model_name} | VAE: {vae_model_name}")

        try:
            pipeline = import_pipeline()
        except ImportError as e:
            print(f"❌ [SeedVR2] Cannot import the pipeline: {e}")
            return empty_result(p, f"Error: {e}")

        memory_plan = build_memory_plan(after_run, blocks_to_swap, swap_io_components)

        # --- Interruption check, passed to the pipeline as a progress callback ---
        def check_interruption(*args, **kwargs):
            # Only .interrupted is checked; the .stopping check was removed on purpose
            if shared.state.interrupted:
                raise SeedVR2Interrupted("User interrupted generation.")

        # --- txt2img: generate the base image first, then upscale it ---
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            if debug_mode: print("[SeedVR2] Generating base image (txt2img)...")

            current_script = None
            if p.scripts is not None:
                for s in p.scripts.scripts:
                    if s.title() == self.title():
                        current_script = s
                        break
                if current_script:
                    p.scripts.scripts.remove(current_script)

            try:
                processed_base = processing.process_images(p)
            finally:
                if current_script and p.scripts:
                    p.scripts.scripts.append(current_script)

            # Only .interrupted is checked (see check_interruption)
            if shared.state.interrupted:
                print("[SeedVR2] Interrupted during Txt2Img generation. Stopping SeedVR2.")
                return processed_base # return the base image as is

            input_img = processed_base.images[0]

            if save_original:
                 images.save_image(input_img, p.outpath_samples, "", processed_base.seed, p.prompt, shared.opts.samples_format, info=processed_base.info, p=p, suffix="-original")

        else:
            # img2img mode: the diffusion pass is skipped, the input image is upscaled directly
            # Only .interrupted is checked (see check_interruption)
            if shared.state.interrupted:
                 return empty_result(p, "Interrupted")
            input_img = p.init_images[0]

        # --- Free VRAM ---
        if unload_sd:
            if debug_mode: print("[SeedVR2] Unloading SD models...")
            modules.sd_models.unload_model_weights()
            devices.torch_gc()

        # --- Resolve the seed ---
        if seed == -1:
            if hasattr(p, 'all_seeds') and p.all_seeds is not None and len(p.all_seeds) > 0 and p.all_seeds[0] != -1:
                actual_seed = int(p.all_seeds[0])
            elif p.seed is not None and p.seed != -1:
                actual_seed = int(p.seed)
            else:
                actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = int(seed)

        if debug_mode:
            print(f"[SeedVR2] Active. DiT: {dit_model_name} | Seed: {actual_seed}")

        # Image preprocessing: PIL -> [1, H, W, C] float16 tensor
        img_tensor = transforms.ToTensor()(input_img).unsqueeze(0).permute(0, 2, 3, 1).to(dtype=torch.float16)

        model_cache = pipeline.get_global_cache()
        dit_path = os.path.join(scan.folder, dit_model_name)
        vae_path = os.path.join(scan.folder, vae_model_name)
        if MODEL_LIFECYCLE.prepare_run(model_cache, dit_path, vae_path, force_reload):
            print("[SeedVR2] Cached models dropped (Force Reload, other files or a previous failed run); loading from disk.")

        succeeded = False
        try:
            debug = pipeline.Debug(enabled=debug_mode)
            check_interruption()

            request = UpscaleRequest(
                dit_model=dit_model_name, vae_model=vae_model_name, model_dir=scan.folder, memory_plan=memory_plan,
                seed=int(actual_seed), resolution=int(resolution), input_noise=float(input_noise),
                latent_noise=float(latent_noise), use_tile_vae=bool(use_tile_vae), tile_size=int(tile_size),
                tile_overlap=int(tile_overlap))
            output_tensor = self._upscale_tensor(pipeline, img_tensor, request, debug, debug_mode, check_interruption)
            succeeded = True

            # Convert the output tensor back to a PIL image
            tensor_f32 = output_tensor[0].cpu().to(dtype=torch.float32)
            tensor_f32 = torch.clamp(tensor_f32, 0, 1)

            out_np = (tensor_f32.numpy() * 255.0).astype(np.uint8)
            out_pil = Image.fromarray(out_np)

            # Save the result
            p.extra_generation_params["SeedVR2 Model"] = dit_model_name
            p.extra_generation_params["SeedVR2 Resolution"] = resolution

            try:
                infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0)
            except:
                infotext = ""

            images.save_image(
                out_pil,
                p.outpath_samples,
                "",
                actual_seed,
                p.prompt,
                shared.opts.samples_format,
                info=infotext,
                p=p,
                suffix="-seedvr2"
            )

            if not debug_mode: print("[SeedVR2] Done.")
            return Processed(p, [out_pil], actual_seed, f"SeedVR2 Upscaled")

        except SeedVR2Interrupted:
            print("\n⏹️ [SeedVR2] Process Interrupted by User.")
            return Processed(p, [input_img], p.seed, "Interrupted")

        except Exception as e:
            print(f"❌ [SeedVR2 Runtime Error]: {e}")
            import traceback
            traceback.print_exc()
            return empty_result(p, f"Runtime Error: {e}")

        finally:
            output_tensor = None
            kept = MODEL_LIFECYCLE.finish_run(model_cache, memory_plan.cache_models, succeeded)
            print("[SeedVR2] Models kept in RAM for the next run." if kept
                  else "[SeedVR2] Models unloaded, memory released.")

    @staticmethod
    def _upscale_tensor(pipeline, img_tensor, request: UpscaleRequest, debug, debug_mode, check_interruption):
        """Run the four SeedVR2 phases on one [1, H, W, C] image tensor and return the pipeline's output tensor.

        The generation context is built fresh every run, so nothing from a failed or interrupted run is reused. After
        each phase the dropped objects are collected and the allocator's cached VRAM is returned, so the next phase
        has the most free VRAM and is less likely to spill into shared memory."""
        ctx = pipeline.setup_generation_context(
            dit_device=devices.device,
            vae_device=devices.device,
            dit_offload_device=request.memory_plan.dit_offload_device,
            vae_offload_device=request.memory_plan.vae_offload_device,
            tensor_offload_device="cpu",
            debug=debug
        )

        runner, cache_context = pipeline.prepare_runner(
            dit_model=request.dit_model,
            vae_model=request.vae_model,
            model_dir=request.model_dir,
            debug=debug,
            ctx=ctx,
            dit_cache=request.memory_plan.cache_models,
            vae_cache=request.memory_plan.cache_models,
            dit_id=DIT_CACHE_ID,
            vae_id=VAE_CACHE_ID,
            block_swap_config=request.memory_plan.block_swap_config(),
            encode_tiled=request.use_tile_vae,
            encode_tile_size=(request.tile_size, request.tile_size),
            encode_tile_overlap=(request.tile_overlap, request.tile_overlap),
            decode_tiled=request.use_tile_vae,
            decode_tile_size=(request.tile_size, request.tile_size),
            decode_tile_overlap=(request.tile_overlap, request.tile_overlap),
            tile_debug="false", attention_mode="sdpa",
            torch_compile_args_dit=None, torch_compile_args_vae=None
        )
        ctx['cache_context'] = cache_context

        frames_tensor, _generation_info = pipeline.compute_generation_info(
            ctx=ctx, images=img_tensor, resolution=request.resolution, max_resolution=0,
            batch_size=1, uniform_batch_size=False, seed=request.seed,
            prepend_frames=0, temporal_overlap=0, debug=debug
        )

        if not debug_mode: print(f"[SeedVR2] Phase 1/4: Encoding {'(Tiled)' if request.use_tile_vae else ''}...")
        ctx = pipeline.encode_all_batches(runner, ctx=ctx, images=frames_tensor, debug=debug, batch_size=1, uniform_batch_size=False, seed=request.seed,
                                          progress_callback=check_interruption,
                                          temporal_overlap=0, resolution=request.resolution, max_resolution=0, input_noise_scale=request.input_noise, color_correction="lab")
        release_freed_memory()

        if not debug_mode: print("[SeedVR2] Phase 2/4: Upscaling...")
        ctx = pipeline.upscale_all_batches(runner, ctx=ctx, debug=debug,
                                           progress_callback=check_interruption,
                                           seed=request.seed, latent_noise_scale=request.latent_noise, cache_model=request.memory_plan.cache_models)
        release_freed_memory()

        if not debug_mode: print(f"[SeedVR2] Phase 3/4: Decoding {'(Tiled)' if request.use_tile_vae else ''}...")
        ctx = pipeline.decode_all_batches(runner, ctx=ctx, debug=debug,
                                          progress_callback=check_interruption,
                                          cache_model=request.memory_plan.cache_models)
        release_freed_memory()

        if not debug_mode: print("[SeedVR2] Phase 4/4: Post-processing...")
        ctx = pipeline.postprocess_all_batches(ctx=ctx, debug=debug,
                                               progress_callback=check_interruption,
                                               color_correction="lab", prepend_frames=0, temporal_overlap=0, batch_size=1)
        return ctx['final_video']

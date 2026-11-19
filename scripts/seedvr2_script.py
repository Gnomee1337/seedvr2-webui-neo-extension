import gc
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

import gradio as gr
from modules import scripts, processing, images, shared, devices
from modules.processing import Processed, StableDiffusionProcessingTxt2Img
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton
import torch
from PIL import Image

# Forge model management (used to unload the SD checkpoint and free VRAM)
import modules.sd_models

# Extension root directory. Forge already puts every extension folder on sys.path, which is what
# makes `seedvr2_neo` and the vendored `seedvr2_vendor` importable; the insert only covers a host that does not.
EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import MAX_BLOCKS_TO_SWAP, AfterRun, build_memory_plan, is_7b_checkpoint, low_vram_warning, \
    parse_flag
from seedvr2_neo.model_discovery import ModelScan, describe_missing_models, find_selection_error, scan_models
from seedvr2_neo.model_lifecycle import ModelLifecycle
from seedvr2_neo.run_inputs import is_unusable_seed, resolve_seed, select_distinct_images, select_generated_images, \
    to_pipeline_image
from seedvr2_neo.run_results import InputDetails, RunImage, SavePlan, describe_generated_inputs, \
    describe_uploaded_inputs, finish_images, merge_run_images, not_upscaled_comment, plan_saving, read_option, \
    seedvr2_generation_params
from seedvr2_neo.upscale_run import INTERRUPTED_MESSAGE, HostHooks, UpscaleRequest, UpscaleRun, import_pipeline
from seedvr2_neo.upscale_settings import NOISE, RESOLUTION, TILE_OVERLAP, TILE_SIZE, USE_TILE_VAE_DEFAULT, Bounds, \
    parse_upscale_settings

# File name suffixes of the saved upscaled image and of the extra copy of the image before the upscale.
RESULT_SUFFIX = "-seedvr2"
ORIGINAL_SUFFIX = "-original"

# Forge options the script reads itself (with the request's `override_settings`).
SAMPLES_SAVE_OPTION = "samples_save"
SAMPLES_FORMAT_OPTION = "samples_format"
SAVE_TO_DIRS_OPTION = "save_to_dirs"


def scan_seedvr2_models() -> ModelScan:
    """Rescan on every call: the user may add files or create the folder while the WebUI runs."""
    return scan_models(shared.models_path, EXTENSION_ROOT)


def slider_range(bounds: Bounds) -> Dict[str, float]:
    """Gradio slider arguments from the bounds `parse_upscale_settings` also enforces on API values."""
    return {"minimum": bounds.minimum, "maximum": bounds.maximum, "step": bounds.step, "value": bounds.default}


def empty_result(p, message: str) -> Processed:
    """A result with no images. `infotexts` is filled because the txt2img ✨ (hires upscale) button reads infotexts[0]."""
    return Processed(p, [], p.seed, message, infotexts=[message])


def release_freed_memory() -> None:
    """Return memory of dropped models and tensors: Python objects first, then the allocator's cached VRAM."""
    gc.collect()
    devices.torch_gc()


# Module-level: Forge creates one Script per tab, while the vendored model cache is shared by the whole process.
MODEL_LIFECYCLE = ModelLifecycle(LoadedModelTracker(), release_freed_memory)


def forge_hooks() -> HostHooks:
    """What the pipeline run takes from Forge, read at each run. Only Interrupt (`.interrupted`) stops a run; "Stop
    generating" (`.stopping_generation`) means "finish the current image", which should not abort the upscale."""
    return HostHooks(device=devices.device, release_memory=release_freed_memory,
                     is_interrupted=lambda: shared.state.interrupted)


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
class RunInputs:
    """The images of one run, already in a mode the pipeline accepts, with what each result takes from its input."""

    images: List[Image.Image]
    details: List[InputDetails]
    seed: int  # the SeedVR2 seed
    result_seed: int  # the seed Forge shows for the whole result


def save_result_image(p, image: Image.Image, index: int, details: InputDetails, infotext: str, suffix: str) -> None:
    """Save one image next to Forge's own outputs, with this request's `override_settings` for the format and date
    subfolders (Forge does not apply them here; see `read_option`)."""
    overrides = getattr(p, "override_settings", None)
    save_to_dirs = read_option(overrides, SAVE_TO_DIRS_OPTION, None)
    try:
        images.save_image(image, p.outpath_samples, "", details.seed, details.prompt,
                          read_option(overrides, SAMPLES_FORMAT_OPTION, shared.opts.samples_format),
                          info=infotext, p=p, suffix=suffix,
                          save_to_dirs=None if save_to_dirs is None else parse_flag(save_to_dirs))
    except Exception as e:  # also other extensions' save callbacks: one failed file must not lose the whole result
        print(f"❌ [SeedVR2] Cannot save image {index + 1}: {type(e).__name__}: {e}")


def save_originals(p, inputs: List[Image.Image], details: List[InputDetails]) -> None:
    """Save the img2img inputs with an `-original` suffix (txt2img base images are saved by Forge itself)."""
    for index, (image, input_details) in enumerate(zip(inputs, details)):
        save_result_image(p, image, index, input_details, input_details.infotext.with_params({}), ORIGINAL_SUFFIX)


def build_result(p, run_images: List[RunImage], details: List[InputDetails], params: Mapping[str, Any],
                 failure: Optional[str], result_seed: int, save_plan: SavePlan) -> Processed:
    """Forge's result: one image and one infotext per input. Upscaled images are saved by the user's settings; when
    some are not upscaled, the result gets a comment saying so."""
    results = finish_images(run_images, details, params, failure, shared.opts.enable_pnginfo)
    for index, result in enumerate(results):
        if result.is_upscaled and save_plan.save_results:
            save_result_image(p, result.image, index, result.details, result.infotext, RESULT_SUFFIX)
    comment = not_upscaled_comment(run_images, failure)
    if comment:
        p.comment(comment)
    infotexts = [result.infotext for result in results]
    return Processed(p, [result.image for result in results], result_seed, infotexts[0], infotexts=infotexts)


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
            resolution = gr.Slider(label="Upscale Resolution (Shortest Edge)", **slider_range(RESOLUTION))

        with gr.Row():
            unload_sd = gr.Checkbox(label="Unload SD Checkpoint", value=True, elem_classes="force-unload",
                                    info="Frees VRAM; the checkpoint reloads on the next generation")
            save_original = gr.Checkbox(label="Save Extra Copy of Original", value=False,
                                        info="img2img: also saves the input image; txt2img images are saved by Forge")
            force_reload = gr.Checkbox(label="Force Reload", value=False)
            debug_mode = gr.Checkbox(label="Show Debug Logs", value=False)

        with gr.Row():
            after_run = gr.Radio(label="After run", choices=[option.value for option in AfterRun],
                                 value=AfterRun.UNLOAD.value,
                                 info="Unload frees VRAM and RAM; Keep in RAM makes the next run start faster but "
                                      "holds RAM the size of the model files")
            blocks_to_swap = gr.Slider(label="BlockSwap: DiT blocks kept in RAM", minimum=0,
                                       maximum=MAX_BLOCKS_TO_SWAP, step=1, value=0)
            swap_io_components = gr.Checkbox(label="BlockSwap: I/O layers too", value=False,
                                             info="Turns BlockSwap on even with 0 blocks; saves little VRAM")

        with gr.Accordion("Advanced Settings (Noise & Tiling)", open=False):
            with gr.Row():
                input_noise = gr.Slider(label="Input Noise Scale", **slider_range(NOISE))
                latent_noise = gr.Slider(label="Latent Noise Scale", **slider_range(NOISE))

            gr.Markdown("ℹ️ **Tiling reduces VRAM usage for huge resolutions.**")
            with gr.Row():
                use_tile_vae = gr.Checkbox(label="Enable VAE Tiling", value=USE_TILE_VAE_DEFAULT)
                # Forge keeps a slider's value in ui-config.json under its label and prefers it to the code default;
                # this label is not "Tile Size" so the 1024 saved by older versions does not override the 512 default.
                # Do not rename it back; any later label change resets the users' saved value again.
                tile_size = gr.Slider(label="VAE Tile Size", **slider_range(TILE_SIZE))
                tile_overlap = gr.Slider(label="Tile Overlap", info="At most half the VAE Tile Size is used",
                                         **slider_range(TILE_OVERLAP))

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
        except Exception as e:  # not only ImportError: the import also probes bf16 on the GPU and may raise (e.g. OOM)
            print(f"❌ [SeedVR2] Cannot import the pipeline: {type(e).__name__}: {e}")
            return empty_result(p, f"Error: cannot import the SeedVR2 pipeline ({type(e).__name__}; see the console)")

        # Read before any generation, so a bad API value costs nothing
        settings = parse_upscale_settings(resolution, input_noise, latent_noise, use_tile_vae, tile_size, tile_overlap)
        memory_plan = build_memory_plan(after_run, blocks_to_swap, swap_io_components)

        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
        collected = self._collect_inputs(p, is_txt2img, seed, debug_mode)
        if isinstance(collected, Processed):
            return collected
        if debug_mode:
            print(f"[SeedVR2] Active. DiT: {dit_model_name} | Seed: {collected.seed} | Images: {len(collected.images)}")

        samples_save = parse_flag(read_option(p.override_settings, SAMPLES_SAVE_OPTION, shared.opts.samples_save))
        save_plan = plan_saving(is_txt2img, samples_save, bool(p.do_not_save_samples), parse_flag(save_original))
        if save_plan.save_originals:
            save_originals(p, collected.images, collected.details)

        # --- Free VRAM ---
        if unload_sd:
            if debug_mode: print("[SeedVR2] Unloading SD models...")
            modules.sd_models.unload_model_weights()
            devices.torch_gc()

        request = UpscaleRequest(dit_model=dit_model_name, vae_model=vae_model_name, model_dir=scan.folder,
                                 memory_plan=memory_plan, seed=collected.seed, settings=settings)
        outcome = UpscaleRun(pipeline, MODEL_LIFECYCLE, forge_hooks()).upscale_images(collected.images, request,
                                                                                       force_reload, debug_mode)

        params = seedvr2_generation_params(dit_model_name, vae_model_name, settings.resolution, collected.seed)
        return build_result(p, merge_run_images(collected.images, outcome.upscaled), collected.details, params,
                            outcome.failure, collected.result_seed, save_plan)

    def _collect_inputs(self, p, is_txt2img: bool, ui_seed, debug_mode) -> Union[Processed, RunInputs]:
        """The images to upscale — every generated image (txt2img) or every distinct input (img2img) — in a mode the
        pipeline accepts, with their details and the seed. A `Processed` instead when there is nothing to upscale."""
        if is_txt2img:
            if debug_mode: print("[SeedVR2] Generating base images (txt2img)...")
            processed_base = self._generate_base_images(p)
            # Only Interrupt stops SeedVR2 (see forge_hooks)
            if shared.state.interrupted:
                print("[SeedVR2] Interrupted during Txt2Img generation. Stopping SeedVR2.")
                return processed_base  # the base images as Forge made them
            generated_count = len(p.all_seeds or [])
            images = select_generated_images(processed_base.images, processed_base.index_of_first_image,
                                             generated_count)
            base_infotexts = select_generated_images(processed_base.infotexts, processed_base.index_of_first_image,
                                                     generated_count)
        else:
            # img2img: the diffusion pass is skipped, the input images are upscaled directly
            if shared.state.interrupted:
                return empty_result(p, INTERRUPTED_MESSAGE)
            images = select_distinct_images(p.init_images or [])
        if not images:
            print("❌ [SeedVR2] There is no image to upscale.")
            return empty_result(p, "SeedVR2: there is no image to upscale")

        if is_unusable_seed(ui_seed):
            print(f"[SeedVR2] Seed {ui_seed!r} is outside 0 … 2^32-1; following the image seed or a random one instead.")
        # img2img never runs process_images, so `p.all_seeds` there is not a seed of these images
        seed = resolve_seed(ui_seed, p.all_seeds if is_txt2img else None, p.seed)
        images = [to_pipeline_image(image) for image in images]
        if is_txt2img:
            details = describe_generated_inputs(p.all_seeds, p.all_prompts, base_infotexts, len(images))
            return RunInputs(images, details, seed, processed_base.seed)
        # Nothing is stored on `p` for img2img: the Batch tab reuses it for the next file
        return RunInputs(images, describe_uploaded_inputs(seed, p.prompt, p.negative_prompt, len(images)), seed, seed)

    def _generate_base_images(self, p) -> Processed:
        """Let Forge generate the txt2img images with every other script applied. This script is taken out of
        `p.scripts` for that call, so Forge does not call its run() again."""
        current_script = None
        if p.scripts is not None:
            for s in p.scripts.scripts:
                if s.title() == self.title():
                    current_script = s
                    break
            if current_script:
                p.scripts.scripts.remove(current_script)

        try:
            return processing.process_images(p)
        finally:
            if current_script and p.scripts:
                p.scripts.scripts.append(current_script)

import gc
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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

from seedvr2_neo.gpu_profiles import MANUAL, describe_profiles, find_profile, profile_choices
from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.memory_plan import MAX_BLOCKS_TO_SWAP, AfterRun, build_memory_plan, is_7b_checkpoint, low_vram_warning, \
    parse_flag
from seedvr2_neo.model_discovery import ModelScan, describe_missing_models, find_selection_error, scan_models
from seedvr2_neo.model_lifecycle import ModelLifecycle
from seedvr2_neo.pipeline_options import DEFAULT_ATTENTION_MODE, DEFAULT_COLOR_CORRECTION, ColorCorrection, \
    PipelineOptions, choice_values, installed_attention_modes, parse_attention_mode, parse_color_correction, \
    resolve_attention_mode
from seedvr2_neo.run_inputs import is_unusable_seed, resolve_seed, select_distinct_images, select_generated_images, \
    to_pipeline_image
from seedvr2_neo.run_results import InputDetails, RunImage, SavePlan, describe_generated_inputs, \
    describe_uploaded_inputs, finish_images, merge_run_images, not_upscaled_comment, plan_saving, read_option, \
    seedvr2_generation_params
from seedvr2_neo.size_preview import DEFAULT_RES_STEP, UNKNOWN_SOURCE_NOTE, describe_output_size, txt2img_final_size
from seedvr2_neo.upscale_run import INTERRUPTED_MESSAGE, HostHooks, UpscaleRequest, UpscaleRun, import_pipeline
from seedvr2_neo.upscale_settings import DEFAULT_SIZE_MODE, FALLBACK_SIZE_MODE, LONG_EDGE, MAX_OUTPUT_LONG_EDGE, NOISE, RESOLUTION, \
    SCALE_FACTOR, TILE_OVERLAP, TILE_SIZE, USE_TILE_VAE_DEFAULT, Bounds, SizeMode, parse_size_mode, \
    parse_upscale_settings

# File name suffixes of the saved upscaled image and of the extra copy of the image before the upscale.
RESULT_SUFFIX = "-seedvr2"
ORIGINAL_SUFFIX = "-original"

# The quick buttons next to the scale factor slider.
QUICK_SCALE_FACTORS = (1.5, 2.0, 4.0)

# Forge's txt2img controls (by elem_id) that decide the size of the image the upscale gets. The Hires. fix flag is the
# hidden checkbox of an InputAccordion, which appends "-checkbox" to its elem_id.
TXT2IMG_SIZE_SOURCES = {
    "txt2img_width": "width", "txt2img_height": "height", "txt2img_hr-checkbox": "enable_hr",
    "txt2img_hr_scale": "hr_scale", "txt2img_hr_resize_x": "hr_resize_x", "txt2img_hr_resize_y": "hr_resize_y",
}
TXT2IMG_SOURCE_ORDER = ("width", "height", "enable_hr", "hr_scale", "hr_resize_x", "hr_resize_y")
# The hidden buttons `javascript/seedvr2.js` presses: once after the page loads in txt2img (Forge applies the values
# saved in ui-config.json after the UI is built, so the line made in `ui()` may name other sizes), and in img2img
# whenever the input image changes size.
TXT2IMG_SIZE_REFRESH_ID = "seedvr2_txt2img_size_refresh"
IMG2IMG_SIZE_REFRESH_ID = "seedvr2_img2img_size_refresh"

# The Script dropdown entry and the API's `script_name` (matched case-insensitively).
SCRIPT_TITLE = "SeedVR2 Upscaler"

# Forge's progress bar is `job_no / job_count` plus the current job's `sampling_step / sampling_steps`
# (`modules/progress.py`); the run moves `sampling_step` through this many steps.
PROGRESS_STEPS = 1000

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


def preview_txt2img_output_size(width, height, enable_hr, hr_scale, hr_resize_x, hr_resize_y, resolution, size_mode,
                                scale_factor, long_edge) -> str:
    """The output size line for txt2img: the upscale of the image Forge will make (after Hires. fix when it is on)."""
    source = txt2img_final_size(width, height, enable_hr, hr_scale, hr_resize_x, hr_resize_y,
                                getattr(shared.opts, "res_step", DEFAULT_RES_STEP))
    return describe_output_size(source, resolution, size_mode, scale_factor, long_edge)


def preview_img2img_output_size(width, height, resolution, size_mode, scale_factor, long_edge) -> str:
    """The output size line for img2img; `width` and `height` come from the browser, 0 when there is no image."""
    return describe_output_size((width, height), resolution, size_mode, scale_factor, long_edge)


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
                     is_interrupted=lambda: shared.state.interrupted, report_progress=report_forge_progress)


def start_forge_progress() -> bool:
    """Give the upscale a job of Forge's progress bar; True when the job counter is Forge's own. After txt2img's
    `process_images` every job is done (the bar reads 100%), and a single img2img run has no job count yet, so the
    upscale becomes the one job. In img2img Batch Forge counts `n_iter` jobs per file, and the counter stays Forge's:
    only the steps of the current job move here, and `finish_forge_progress` moves past the file. `job_count` is never
    raised above 1 here: Forge's Interrupt button behaves differently for several jobs."""
    state = shared.state
    job_count, job_no = read_state_count(state.job_count), read_state_count(state.job_no)
    forge_counts_jobs = 0 <= job_no < job_count
    if not forge_counts_jobs:
        state.job_count, state.job_no = 1, 0
    state.sampling_steps, state.sampling_step = PROGRESS_STEPS, 0
    return forge_counts_jobs


def finish_forge_progress(forge_counts_jobs: bool, iterations: Any) -> None:
    """In img2img Batch, move Forge's job counter past this file: Forge leaves that to `process_images` (`nextjob()`
    per iteration), which this script replaces in img2img, so without it every file would restart the same part of
    the bar."""
    if not forge_counts_jobs:
        return
    state = shared.state
    job_count, job_no = read_state_count(state.job_count), read_state_count(state.job_no)
    state.job_no = min(job_count, job_no + max(1, read_state_count(iterations)))
    state.sampling_step = 0


def read_state_count(value: Any) -> int:
    """A job counter of `shared.state` (or `p.n_iter`) as an int; 0 when it is not a number."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def report_forge_progress(fraction: float, text: str) -> None:
    """Move Forge's progress bar to `fraction` (0…1, clamped) of the current job and show `text` next to it; a
    fraction that is not a number (NaN) leaves the bar where it is."""
    if isinstance(fraction, (int, float)) and not math.isnan(fraction):
        shared.state.sampling_step = int(round(min(max(fraction, 0.0), 1.0) * PROGRESS_STEPS))
    shared.state.textinfo = text


def get_total_vram_bytes() -> Optional[int]:
    """Total memory of Forge's GPU, or None on a non-CUDA device or when it cannot be read."""
    if devices.device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(devices.device).total_memory)
    except (RuntimeError, AssertionError) as e:
        print(f"[SeedVR2] Cannot read the GPU memory size: {e}")
        return None


def selected_is_7b(scan: ModelScan, dit_model_name: Optional[str]) -> Optional[bool]:
    """Whether the selected DiT is 7B, or None when no listed DiT is selected."""
    if not dit_model_name or scan.folder is None or dit_model_name not in scan.dit_models:
        return None
    return is_7b_checkpoint(dit_model_name, os.path.join(scan.folder, dit_model_name))


def describe_vram_warning(scan: ModelScan, dit_model_name: Optional[str]) -> str:
    """The 7B-on-a-small-GPU note for the selected DiT, or an empty string."""
    is_7b = selected_is_7b(scan, dit_model_name)
    if is_7b is None:
        return ""
    return low_vram_warning(is_7b, get_total_vram_bytes()) or ""


def apply_gpu_profile(profile_label: Any, dit_model_name: Optional[str]) -> Tuple[Dict[str, Any], ...]:
    """The memory sliders (BlockSwap, VAE tile size, tile overlap) for the chosen GPU profile and DiT; Manual and an
    unknown profile leave them as they are."""
    profile = find_profile(profile_label)
    if profile is None:
        return gr.update(), gr.update(), gr.update()
    is_7b = bool(selected_is_7b(scan_seedvr2_models(), dit_model_name))
    return (gr.update(value=profile.blocks_to_swap(is_7b)), gr.update(value=profile.tile_size),
            gr.update(value=profile.tile_overlap))


def resolve_pipeline_options(pipeline: Any, color_correction: Any, attention_mode: Any) -> PipelineOptions:
    """The run's color correction and attention backend from raw UI / API values; a backend the pipeline could not
    import falls back to `sdpa` with a console note."""
    attention, note = resolve_attention_mode(parse_attention_mode(attention_mode),
                                             getattr(pipeline, "attention_available", {}))
    if note:
        print(f"[SeedVR2] {note}")
    return PipelineOptions(parse_color_correction(color_correction), attention)


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


def build_result(p, run_images: List[RunImage], details: List[InputDetails], params: List[Mapping[str, Any]],
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
    def __init__(self):
        super().__init__()
        # Forge's txt2img size controls by name (`TXT2IMG_SIZE_SOURCES`), seen in `after_component`.
        self._txt2img_size_sources: Dict[str, Any] = {}

    def title(self):
        return SCRIPT_TITLE

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        # Layout only: Forge keys saved values (ui-config.json) by label, not by the rows or accordions around a
        # control, and run() gets the controls in the order of the list returned below, not in the order shown.
        dit_model, vae_model = self._model_controls()
        size_mode, resolution, scale_factor, long_edge = self._size_controls(is_img2img)
        after_run, unload_sd, gpu_profile, blocks_to_swap, swap_io_components = self._memory_controls()
        (seed, save_original, force_reload, debug_mode, input_noise, latent_noise, use_tile_vae, tile_size,
         tile_overlap, color_correction, attention_mode) = self._advanced_controls()
        self._connect_gpu_profile(gpu_profile, dit_model, [blocks_to_swap, tile_size, tile_overlap])
        # The GPU profile is not returned: it only fills the sliders, so a run never depends on it
        return [dit_model, vae_model, seed, resolution, input_noise, latent_noise, force_reload, unload_sd, save_original,
                use_tile_vae, tile_size, tile_overlap, debug_mode, after_run, blocks_to_swap, swap_io_components,
                size_mode, scale_factor, long_edge, color_correction, attention_mode]

    def _model_controls(self):
        """The DiT and VAE dropdowns with the rescan button and the notes under them."""
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
        return dit_model, vae_model

    def _size_controls(self, is_img2img):
        """The output size group: the mode, the control of each mode (only the chosen one shown) and the size line."""
        with gr.Group(elem_classes=["seedvr2-section"]):
            size_mode = gr.Radio(label="Output Size Mode", choices=[mode.value for mode in SizeMode],
                                 value=DEFAULT_SIZE_MODE.value,
                                 info=f"Scale factor and Long edge keep the long edge at most {MAX_OUTPUT_LONG_EDGE} "
                                      f"px; Scale factor raises small images towards {RESOLUTION.minimum} px")
            # Forge would apply a mode saved in ui-config.json after the UI is built, while the visible slider column
            # comes from the code default: the radio and the slider would disagree. So the mode always starts at the
            # code default and is never saved; the sliders keep their saved values.
            size_mode.do_not_save_to_config = True
            # Only the active mode's control is shown. The shortest edge slider keeps its old label so the values
            # users saved still apply.
            with gr.Column(visible=DEFAULT_SIZE_MODE is SizeMode.SCALE_FACTOR) as scale_factor_controls:
                with gr.Row(elem_classes=["seedvr2-quick-scale-row"]):
                    scale_factor = gr.Slider(label="Scale Factor", scale=1, **slider_range(SCALE_FACTOR))
                    for factor in QUICK_SCALE_FACTORS:
                        gr.Button(value=f"×{factor:g}", size="sm", scale=0, min_width=56,
                                  elem_classes=["seedvr2-quick-scale"]).click(fn=lambda factor=factor: factor,
                                                                              inputs=[], outputs=[scale_factor])
            with gr.Column(visible=DEFAULT_SIZE_MODE is SizeMode.SHORTEST_EDGE) as shortest_edge_controls:
                resolution = gr.Slider(label="Upscale Resolution (Shortest Edge)", **slider_range(RESOLUTION))
            with gr.Column(visible=DEFAULT_SIZE_MODE is SizeMode.LONG_EDGE) as long_edge_controls:
                long_edge = gr.Slider(label="Output Long Edge (px)", **slider_range(LONG_EDGE))
            size_controls = [resolution, size_mode, scale_factor, long_edge]
            txt2img_inputs = None if is_img2img else self._txt2img_preview_inputs(size_controls)
            output_size_note = gr.Markdown(
                UNKNOWN_SOURCE_NOTE if txt2img_inputs is None
                else preview_txt2img_output_size(*(component.value for component in txt2img_inputs)),
                elem_classes=["seedvr2-size-note"])
        mode_controls = {SizeMode.SCALE_FACTOR: scale_factor_controls, SizeMode.SHORTEST_EDGE: shortest_edge_controls,
                         SizeMode.LONG_EDGE: long_edge_controls}
        size_mode.change(fn=self._show_size_controls, inputs=[size_mode],
                         outputs=[mode_controls[mode] for mode in SizeMode])
        if is_img2img:
            self._preview_img2img_output_size(output_size_note, size_controls)
        elif txt2img_inputs is not None:
            self._preview_txt2img_output_size(output_size_note, txt2img_inputs)
        return size_mode, resolution, scale_factor, long_edge

    @staticmethod
    def _memory_controls():
        """After run and the SD unload, then the GPU profile and BlockSwap: on the main screen, since 7B on a small
        GPU needs them."""
        with gr.Row():
            after_run = gr.Radio(label="After run", choices=[option.value for option in AfterRun],
                                 value=AfterRun.UNLOAD.value,
                                 info="Unload frees VRAM and RAM; Keep in RAM makes the next run start faster but "
                                      "holds RAM the size of the model files")
            unload_sd = gr.Checkbox(label="Unload SD Checkpoint", value=True, elem_classes="force-unload",
                                    info="Frees VRAM; the checkpoint reloads on the next generation")
        with gr.Row():
            gpu_profile = gr.Dropdown(label="GPU Profile", choices=profile_choices(), value=MANUAL,
                                      info=describe_profiles())
            # Always starts at Manual: a profile saved in ui-config.json would be shown next to slider values the
            # user changed afterwards.
            gpu_profile.do_not_save_to_config = True
            blocks_to_swap = gr.Slider(label="BlockSwap: DiT blocks kept in RAM", minimum=0,
                                       maximum=MAX_BLOCKS_TO_SWAP, step=1, value=0)
            swap_io_components = gr.Checkbox(label="BlockSwap: I/O layers too", value=False,
                                             info="Turns BlockSwap on even with 0 blocks; saves little VRAM")
        return after_run, unload_sd, gpu_profile, blocks_to_swap, swap_io_components

    @staticmethod
    def _advanced_controls():
        """The settings changed rarely: seed, extra saving, reload and logs, noise, VAE tiling, color correction and
        attention."""
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                seed = gr.Number(label="Seed (-1 = Random/Follow)", value=-1, precision=0)
                save_original = gr.Checkbox(label="Save Extra Copy of Original", value=False,
                                            info="img2img: also saves the input image; txt2img images are saved by "
                                                 "Forge")
                force_reload = gr.Checkbox(label="Force Reload", value=False,
                                           info="Reads the model files again; matters only with Keep in RAM")
                debug_mode = gr.Checkbox(label="Show Debug Logs", value=False)
            with gr.Row():
                input_noise = gr.Slider(label="Input Noise Scale", **slider_range(NOISE))
                latent_noise = gr.Slider(label="Latent Noise Scale", **slider_range(NOISE))
            with gr.Row():
                use_tile_vae = gr.Checkbox(label="Enable VAE Tiling", value=USE_TILE_VAE_DEFAULT,
                                           info="Less VRAM for large images")
                # Forge keeps a slider's value in ui-config.json under its label and prefers it to the code default;
                # this label is not "Tile Size" so the 1024 saved by older versions does not override the 512 default.
                # Do not rename it back; any later label change resets the users' saved value again.
                tile_size = gr.Slider(label="VAE Tile Size", **slider_range(TILE_SIZE))
                tile_overlap = gr.Slider(label="Tile Overlap", info="At most half the VAE Tile Size is used",
                                         **slider_range(TILE_OVERLAP))
            with gr.Row():
                color_correction = gr.Dropdown(label="Color Correction", choices=choice_values(ColorCorrection),
                                               value=DEFAULT_COLOR_CORRECTION.value,
                                               info="Matches the result's colors to the input; lab keeps them closest")
                attention_mode = gr.Dropdown(label="Attention", choices=choice_values(installed_attention_modes()),
                                             value=DEFAULT_ATTENTION_MODE.value,
                                             info="sdpa is built into PyTorch; others appear when their package is "
                                                  "installed")
        return (seed, save_original, force_reload, debug_mode, input_noise, latent_noise, use_tile_vae, tile_size,
                tile_overlap, color_correction, attention_mode)

    @staticmethod
    def _connect_gpu_profile(gpu_profile, dit_model, sliders) -> None:
        """A chosen profile fills `sliders` (BlockSwap, VAE tile size, tile overlap), again when the DiT changes
        between 3B and 7B; moving one of them by hand sets the profile back to Manual. `.input` fires only for the
        user's own changes, not for the profile's."""
        event = dict(fn=apply_gpu_profile, inputs=[gpu_profile, dit_model], outputs=sliders, show_progress="hidden")
        gpu_profile.change(**event)
        dit_model.change(**event)
        for slider in sliders:
            slider.input(fn=lambda: MANUAL, inputs=[], outputs=[gpu_profile], show_progress="hidden")

    def after_component(self, component, **kwargs):
        """Keep the txt2img controls the output size preview reads; Forge calls this for every component it builds,
        and builds them before the scripts' UI."""
        elem_id = kwargs.get("elem_id")
        name = TXT2IMG_SIZE_SOURCES.get(elem_id) if isinstance(elem_id, str) else None
        if name:
            self._txt2img_size_sources[name] = component

    def _txt2img_preview_inputs(self, size_controls) -> Optional[List[Any]]:
        """Forge's size controls followed by ours, in the order `preview_txt2img_output_size` takes them; None when
        one of Forge's was not seen (a UI layout setting can build the scripts first), so the note stays generic."""
        sources = self._txt2img_size_sources
        if any(name not in sources for name in TXT2IMG_SOURCE_ORDER):
            return None
        return [sources[name] for name in TXT2IMG_SOURCE_ORDER] + size_controls

    @staticmethod
    def _preview_txt2img_output_size(note, inputs) -> None:
        """Update `note` whenever the image size (width, height, Hires. fix) or a size control changes, and once after
        the page loads, through the hidden refresh button `javascript/seedvr2.js` presses."""
        event = dict(fn=preview_txt2img_output_size, inputs=inputs, outputs=[note], show_progress="hidden")
        refresh = gr.Button(value="Refresh output size", elem_id=TXT2IMG_SIZE_REFRESH_ID,
                            elem_classes=["seedvr2-hidden"])
        refresh.click(**event)
        for component in inputs:
            component.change(**event)

    @staticmethod
    def _preview_img2img_output_size(note, size_controls) -> None:
        """Update `note` when a size control changes or the input image does. The image size is read in the browser
        (`javascript/seedvr2.js`), so the image itself is not sent to the server; the script's JS presses the hidden
        refresh button when the visible input image changes size."""
        source_size = [gr.Number(value=0, visible=False), gr.Number(value=0, visible=False)]
        refresh = gr.Button(value="Refresh output size", elem_id=IMG2IMG_SIZE_REFRESH_ID,
                            elem_classes=["seedvr2-hidden"])
        event = dict(fn=preview_img2img_output_size, _js="seedvr2Img2imgPreview", inputs=source_size + size_controls,
                     outputs=[note], show_progress="hidden")
        refresh.click(**event)
        for component in size_controls:
            component.change(**event)

    @staticmethod
    def _show_size_controls(size_mode):
        """Show the control of the chosen size mode, hide the other two."""
        chosen = parse_size_mode(size_mode)
        return tuple(gr.update(visible=chosen is mode) for mode in SizeMode)

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
            blocks_to_swap=0, swap_io_components=False, size_mode=FALLBACK_SIZE_MODE.value,
            scale_factor=SCALE_FACTOR.default, long_edge=LONG_EDGE.default,
            color_correction=DEFAULT_COLOR_CORRECTION.value, attention_mode=DEFAULT_ATTENTION_MODE.value):
        # An API caller may send fewer values than there are controls (older callers send 13, 16 or 19). Forge does not
        # pad the list with the UI's values: these defaults, or whatever Forge shifts into the missing positions, are
        # read leniently, and an unreadable size mode means the shortest edge, the only mode older callers know.
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
        settings = parse_upscale_settings(resolution, input_noise, latent_noise, use_tile_vae, tile_size, tile_overlap,
                                          size_mode, scale_factor, long_edge)
        memory_plan = build_memory_plan(after_run, blocks_to_swap, swap_io_components)
        options = resolve_pipeline_options(pipeline, color_correction, attention_mode)

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
                                 memory_plan=memory_plan, seed=collected.seed, settings=settings, options=options)
        forge_counts_jobs = start_forge_progress()
        outcome = UpscaleRun(pipeline, MODEL_LIFECYCLE, forge_hooks()).upscale_images(collected.images, request,
                                                                                       force_reload, debug_mode)
        finish_forge_progress(forge_counts_jobs, getattr(p, "n_iter", 1))

        params = [seedvr2_generation_params(dit_model_name, vae_model_name, settings, image.size, collected.seed,
                                            options)
                  for image in collected.images]
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

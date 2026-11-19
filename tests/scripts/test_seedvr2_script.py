"""The Forge script `scripts/seedvr2_script.py`, run through `Script.run()` against the stubs of `forge_stubs.py`.

These tests describe what users see: which images come back, with which infotext, what is saved, what happens to the
SD checkpoint and the SeedVR2 models, and how errors and interrupts end a run.
"""

import ast
import inspect
import os
import sys

import pytest
from PIL import Image

from forge_stubs import DIT, FORGE_DEVICE, PIPELINE_FUNCTIONS, UPSCALE_FACTOR, VAE, base_image_color, make_img2img, \
    run_args

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

UPSCALED_SIZE = (8 * UPSCALE_FACTOR, 8 * UPSCALE_FACTOR)
CACHE_IDS = ["forge_dit", "forge_vae"]


def run(script, p, **overrides):
    return script.run(p, **run_args(**overrides))


def corner_color(image):
    return image.convert("RGB").getpixel((0, 0))


def colors_match(actual, expected, tolerance=2):
    """The pipeline works in float16 and the run truncates to 8 bits, so a channel may drift by a level or two."""
    return len(actual) == len(expected) and all(
        abs(a - e) <= tolerance for actual_color, expected_color in zip(actual, expected)
        for a, e in zip(actual_color, expected_color))


# --- UI --------------------------------------------------------------------------------------------------------------

def test_ui_returns_one_component_per_run_argument(script):
    components = script.ui(is_img2img=False)
    run_parameters = list(inspect.signature(script.run).parameters)[1:]  # without `p`
    assert len(components) == len(run_parameters)
    assert list(run_args()) == run_parameters


# What each `Script.run` parameter is, by the label of the component `ui()` returns for it.
RUN_PARAMETER_LABELS = {
    "dit_model_name": "DiT Model", "vae_model_name": "VAE Model", "seed": "Seed (-1 = Random/Follow)",
    "resolution": "Upscale Resolution (Shortest Edge)", "input_noise": "Input Noise Scale",
    "latent_noise": "Latent Noise Scale", "force_reload": "Force Reload", "unload_sd": "Unload SD Checkpoint",
    "save_original": "Save Extra Copy of Original", "use_tile_vae": "Enable VAE Tiling", "tile_size": "VAE Tile Size",
    "tile_overlap": "Tile Overlap", "debug_mode": "Show Debug Logs", "after_run": "After run",
    "blocks_to_swap": "BlockSwap: DiT blocks kept in RAM", "swap_io_components": "BlockSwap: I/O layers too",
    "size_mode": "Output Size Mode", "scale_factor": "Scale Factor", "long_edge": "Output Long Edge (px)",
}


def test_ui_components_come_in_the_order_of_the_run_parameters(script):
    """Forge passes the UI values to `run()` by position; a reordered control would feed the wrong parameter."""
    labels = [component.kwargs["label"] for component in script.ui(is_img2img=False)]
    run_parameters = list(inspect.signature(script.run).parameters)[1:]
    assert labels == [RUN_PARAMETER_LABELS[name] for name in run_parameters]


def test_vae_tile_slider_defaults_to_512_under_a_new_label(script):
    """A new label is a new ui-config.json key, so the 1024 that older versions saved under "Tile Size" no longer
    applies; with 512 the VAE phases finish on 8 GB of VRAM."""
    tile_size = next(component for component in script.ui(is_img2img=True)
                     if component.kwargs["label"] == "VAE Tile Size")
    assert (tile_size.kwargs["value"], tile_size.kwargs["minimum"], tile_size.kwargs["maximum"]) == (512, 512, 2048)


def test_ui_offers_the_models_in_the_folder(script):
    dit_model, vae_model = script.ui(is_img2img=True)[:2]
    assert dit_model.kwargs["choices"] == [DIT] and dit_model.kwargs["value"] == DIT
    assert vae_model.kwargs["choices"] == [VAE] and vae_model.kwargs["value"] == VAE


def test_refresh_keeps_the_selection_while_the_files_exist(script):
    dit_update, vae_update, note_update = script._refresh_models(DIT, VAE)
    assert dit_update["value"] == DIT and vae_update["value"] == VAE
    assert note_update["visible"] is False


def test_refresh_reports_an_emptied_folder(script, forge, tmp_path):
    for model in (tmp_path / "models" / "SeedVR2").iterdir():
        model.unlink()
    dit_update, vae_update, note_update = script._refresh_models(DIT, VAE)
    assert dit_update["choices"] == [] and dit_update["value"] is None
    assert note_update["visible"] is True and note_update["value"]


def test_script_is_shown_on_both_tabs(script):
    assert script.show(False) and script.show(True)


# --- txt2img -------------------------------------------------------------------------------------------------------

def test_txt2img_upscales_every_generated_image_and_drops_the_grid(script, make_txt2img):
    p = make_txt2img(batch_size=2, n_iter=2)
    processed = run(script, p)
    assert len(processed.images) == 4 and len(processed.infotexts) == 4
    assert [image.size for image in processed.images] == [UPSCALED_SIZE] * 4
    assert colors_match([corner_color(image) for image in processed.images], [base_image_color(i) for i in range(4)])
    assert processed.index_of_first_image == 0


def test_txt2img_infotext_keeps_forge_parameters_and_adds_seedvr2_keys(script, make_txt2img):
    processed = run(script, make_txt2img(batch_size=2), seed=7)
    for index, infotext in enumerate(processed.infotexts):
        prompt, params = infotext.rsplit("\n", 1)
        assert prompt == "a cat"
        assert params.startswith(f"Steps: 20, Seed: {1000 + index}, Size: 8x8, ")
        assert f"SeedVR2 Model: {DIT}, SeedVR2 VAE: {VAE}, SeedVR2 Resolution: 1080, SeedVR2 Seed: 7" in params
        assert processed.images[index].info["parameters"] == infotext
    assert processed.seed == 1000


def test_txt2img_seed_follows_the_first_image_when_not_chosen(script, make_txt2img, pipeline):
    processed = run(script, make_txt2img(seed=555))
    assert pipeline.generation_kwargs[0]["seed"] == 555
    assert "SeedVR2 Seed: 555" in processed.infotexts[0]


def test_txt2img_generates_without_itself_and_puts_itself_back(script, make_txt2img, forge):
    p = make_txt2img()
    scripts_before = list(p.scripts.scripts)
    run(script, p)
    assert forge.process_images_calls == 1
    assert [s.title() for s in forge.scripts_during_generation[0]] == ["Other script"]
    assert [s.title() for s in p.scripts.scripts] == [s.title() for s in scripts_before]


def test_txt2img_interrupted_while_generating_returns_the_base_images(script, make_txt2img, forge, pipeline):
    forge.interrupt_generation = True
    processed = run(script, make_txt2img(batch_size=2))
    assert len(processed.images) == 3  # the grid and both images, as Forge made them
    assert processed.index_of_first_image == 1
    assert pipeline.passes == [] and forge.sd_unloads == 0


def test_txt2img_saves_upscaled_images_with_their_own_seed_and_infotext(script, make_txt2img, forge):
    processed = run(script, make_txt2img(batch_size=2))
    assert [saved.suffix for saved in forge.saved] == ["-seedvr2", "-seedvr2"]
    assert [saved.seed for saved in forge.saved] == [1000, 1001]
    assert [saved.info for saved in forge.saved] == processed.infotexts
    assert [saved.extension for saved in forge.saved] == ["png", "png"]
    assert processed.images[0].already_saved_as


def test_txt2img_ignores_save_original(script, make_txt2img, forge):
    run(script, make_txt2img(), save_original=True)
    assert [saved.suffix for saved in forge.saved] == ["-seedvr2"]


def test_saving_uses_the_request_format_and_subfolder_overrides(script, make_txt2img, forge):
    run(script, make_txt2img(override_settings={"samples_format": "jpg", "save_to_dirs": False}))
    assert forge.saved[0].extension == "jpg" and forge.saved[0].save_to_dirs is False


@pytest.mark.parametrize("setup", ["option_off", "request_override_off", "caller_disabled_saving"])
def test_nothing_is_saved_when_saving_is_off(script, make_txt2img, forge, setup):
    fields = {}
    if setup == "option_off":
        forge.opts.samples_save = False
    elif setup == "request_override_off":
        fields["override_settings"] = {"samples_save": False}
    else:
        fields["do_not_save_samples"] = True
    processed = run(script, make_txt2img(**fields))
    assert forge.saved == []
    assert processed.images[0].size == UPSCALED_SIZE


def test_a_failing_save_keeps_the_result_and_the_other_files(script, make_txt2img, forge):
    forge.save_error, forge.failing_saves = OSError("disk full"), 1
    processed = run(script, make_txt2img(batch_size=2))
    assert [image.size for image in processed.images] == [UPSCALED_SIZE] * 2
    assert [saved.seed for saved in forge.saved] == [1001]  # the second image is saved after the first one failed


def test_infotext_is_not_embedded_when_png_info_is_off(script, make_txt2img, forge):
    forge.opts.enable_pnginfo = False
    processed = run(script, make_txt2img())
    assert "parameters" not in processed.images[0].info
    assert "SeedVR2 Model" in processed.infotexts[0]


# --- img2img -------------------------------------------------------------------------------------------------------

def test_img2img_upscales_each_distinct_input_once(script, pipeline):
    small = Image.new("RGB", (8, 8), (255, 0, 0))
    wide = Image.new("RGB", (16, 8), (0, 0, 255))
    processed = run(script, make_img2img([small, small, wide]))  # the Batch tab repeats one file
    assert [image.size for image in processed.images] == [UPSCALED_SIZE, (32, 16)]
    assert [tuple(tensor.shape) for tensor in pipeline.passes] == [(1, 8, 8, 3), (1, 8, 16, 3)]


def test_img2img_images_of_one_size_go_through_one_pass(script, pipeline):
    inputs = [Image.new("RGB", (8, 8), (index * 50, 0, 0)) for index in range(3)]
    processed = run(script, make_img2img(inputs))
    assert [tuple(tensor.shape) for tensor in pipeline.passes] == [(3, 8, 8, 3)]
    assert colors_match([corner_color(image) for image in processed.images], [(index * 50, 0, 0) for index in range(3)])


@pytest.mark.parametrize("mode, expected_mode", [("L", "RGB"), ("P", "RGB"), ("I;16", "RGB"), ("CMYK", "RGB"),
                                                 ("LA", "RGBA"), ("RGBA", "RGBA")])
def test_img2img_any_image_mode_comes_back_as_rgb_or_rgba(script, mode, expected_mode):
    processed = run(script, make_img2img([Image.new(mode, (8, 8))]))
    assert processed.images[0].mode == expected_mode and processed.images[0].size == UPSCALED_SIZE


def test_img2img_uses_the_ui_seed_and_the_prompt(script, pipeline):
    processed = run(script, make_img2img([Image.new("RGB", (8, 8))]), seed=42)
    assert pipeline.generation_kwargs[0]["seed"] == 42
    assert processed.seed == 42
    assert processed.infotexts[0].startswith("a dog\n")
    assert "SeedVR2 Seed: 42" in processed.infotexts[0]


def test_img2img_saves_the_original_when_asked(script, forge):
    run(script, make_img2img([Image.new("RGB", (8, 8))]), save_original=True)
    assert [saved.suffix for saved in forge.saved] == ["-original", "-seedvr2"]
    assert forge.saved[0].image.size == (8, 8)
    assert forge.saved[0].info == "a dog"  # the input's own infotext: the prompt, no SeedVR2 parameters


def test_img2img_saves_no_original_when_saving_is_off(script, forge):
    forge.opts.samples_save = False
    run(script, make_img2img([Image.new("RGB", (8, 8))]), save_original=True)
    assert forge.saved == []


@pytest.mark.parametrize("init_images", [None, []])
def test_img2img_without_inputs_returns_an_empty_result(script, forge, pipeline, init_images):
    processed = run(script, make_img2img(init_images))
    assert processed.images == []
    assert processed.infotexts == ["SeedVR2: there is no image to upscale"]
    assert forge.sd_unloads == 0 and pipeline.passes == []


def test_img2img_interrupted_before_the_start_returns_an_empty_result(script, forge, pipeline):
    forge.state.interrupted = True
    processed = run(script, make_img2img([Image.new("RGB", (8, 8))]))
    assert processed.images == [] and processed.info == "Interrupted"
    assert pipeline.passes == []


# --- failures and interrupts -----------------------------------------------------------------------------------------

def test_missing_model_stops_before_any_generation(script, make_txt2img, forge, pipeline):
    processed = run(script, make_txt2img(), dit_model_name="gone.safetensors")
    assert processed.images == [] and processed.info.startswith("SeedVR2: ")
    assert forge.process_images_calls == 0 and forge.sd_unloads == 0 and pipeline.passes == []


def test_broken_pipeline_install_stops_before_any_generation(script, make_txt2img, forge, monkeypatch):
    monkeypatch.setitem(sys.modules, "seedvr2_vendor.core.generation_phases", None)  # makes its import fail
    processed = run(script, make_txt2img())
    assert processed.images == []
    assert processed.info == "Error: cannot import the SeedVR2 pipeline (ModuleNotFoundError; see the console)"
    assert forge.process_images_calls == 0 and forge.sd_unloads == 0


@pytest.mark.parametrize("phase", ["encode", "upscale", "decode", "postprocess"])
def test_pipeline_error_returns_the_inputs_with_the_error_type_only(script, make_txt2img, forge, pipeline, phase):
    pipeline.fail_in = phase
    processed = run(script, make_txt2img(batch_size=2))
    assert [image.size for image in processed.images] == [(8, 8), (8, 8)]
    for infotext in processed.infotexts:
        assert 'SeedVR2 Error: "Runtime Error: RuntimeError"' in infotext
        assert "Users" not in infotext  # the exception text may hold local paths; only its type is shown
    assert any("2 of 2 images returned without upscaling" in comment for comment in processed.comments.splitlines())
    assert forge.saved == []


def test_error_in_a_later_pass_keeps_the_images_already_upscaled(script, pipeline):
    pipeline.fail_in, pipeline.only_in_pass = "upscale", 1
    inputs = [Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))]
    processed = run(script, make_img2img(inputs))
    assert [image.size for image in processed.images] == [UPSCALED_SIZE, (16, 8)]
    assert "SeedVR2 Error" not in processed.infotexts[0] and "SeedVR2 Error" in processed.infotexts[1]
    assert "1 of 2 images returned without upscaling" in processed.comments


def test_interrupt_in_a_later_pass_keeps_the_images_already_upscaled(script, pipeline):
    pipeline.interrupt_in, pipeline.only_in_pass = "encode", 1
    processed = run(script, make_img2img([Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))]))
    assert [image.size for image in processed.images] == [UPSCALED_SIZE, (16, 8)]
    assert "SeedVR2 Error: Interrupted" in processed.infotexts[1] and "SeedVR2 Error" not in processed.infotexts[0]


def test_interrupt_during_a_phase_returns_the_inputs_marked_interrupted(script, make_txt2img, forge, pipeline):
    pipeline.interrupt_in = "decode"
    processed = run(script, make_txt2img())
    assert processed.images[0].size == (8, 8)
    assert "SeedVR2 Error: Interrupted" in processed.infotexts[0]
    assert "postprocess" not in pipeline.phases_run


def test_returned_input_is_a_copy_that_keeps_the_file_forge_saved(script, pipeline):
    pipeline.fail_in = "upscale"
    original = Image.new("RGB", (8, 8))
    original.already_saved_as = "outputs/00001.png"
    processed = run(script, make_img2img([original]))
    returned = processed.images[0]
    assert returned is not original and "parameters" not in original.info
    assert returned.already_saved_as == "outputs/00001.png"
    assert "SeedVR2 Error" in returned.info["parameters"]


# --- SD checkpoint and SeedVR2 models ----------------------------------------------------------------------------------

def test_sd_checkpoint_is_unloaded_only_when_asked(script, make_txt2img, forge):
    run(script, make_txt2img(), unload_sd=False)
    assert forge.sd_unloads == 0
    run(script, make_txt2img(), unload_sd=True)
    assert forge.sd_unloads == 1


def test_models_are_unloaded_after_a_run_by_default(script, make_txt2img, pipeline):
    run(script, make_txt2img())
    assert pipeline.runner_kwargs[0]["dit_cache"] is False and pipeline.runner_kwargs[0]["vae_cache"] is False
    assert pipeline.cache.removed == CACHE_IDS * 2  # dropped before the run (nothing trusted yet) and after it


def test_the_run_uses_forge_device_and_releases_memory(script, make_txt2img, forge, pipeline):
    run(script, make_txt2img(), unload_sd=False)
    context = pipeline.context_kwargs[0]
    assert context["dit_device"] == FORGE_DEVICE and context["vae_device"] == FORGE_DEVICE
    releases_without_sd_unload = forge.torch_gc_calls
    assert releases_without_sd_unload >= 3  # after encode, upscale and decode
    run(script, make_txt2img(), unload_sd=True)
    assert forge.torch_gc_calls - releases_without_sd_unload == releases_without_sd_unload + 1  # + after the SD unload


@pytest.mark.parametrize("debug_mode", [False, True])
def test_show_debug_logs_reaches_the_pipeline_logger(script, make_txt2img, pipeline, debug_mode):
    run(script, make_txt2img(), debug_mode=debug_mode)
    assert pipeline.context_kwargs[0]["debug"].enabled is debug_mode


def test_keep_in_ram_reuses_the_models_on_the_next_run(script, make_txt2img, pipeline):
    run(script, make_txt2img(), after_run="Keep in RAM")
    assert pipeline.cache.cached == set(CACHE_IDS)
    removed_before = list(pipeline.cache.removed)
    run(script, make_txt2img(), after_run="Keep in RAM")
    assert pipeline.cache.removed == removed_before  # nothing dropped: same files
    assert pipeline.cache.cached == set(CACHE_IDS)


def test_force_reload_drops_kept_models(script, make_txt2img, pipeline):
    run(script, make_txt2img(), after_run="Keep in RAM")
    removed_before = len(pipeline.cache.removed)
    run(script, make_txt2img(), after_run="Keep in RAM", force_reload=True)
    assert pipeline.cache.removed[removed_before:removed_before + 2] == CACHE_IDS


def test_models_are_dropped_after_an_error_even_with_keep_in_ram(script, make_txt2img, pipeline):
    pipeline.fail_in = "decode"
    run(script, make_txt2img(), after_run="Keep in RAM")
    assert pipeline.cache.cached == set()


def test_block_swap_settings_reach_the_pipeline(script, make_txt2img, pipeline):
    run(script, make_txt2img(), blocks_to_swap=10, swap_io_components=True)
    config = pipeline.runner_kwargs[0]["block_swap_config"]
    assert config["blocks_to_swap"] == 10 and config["swap_io_components"] is True


# --- settings from the API -------------------------------------------------------------------------------------------

def test_api_values_outside_the_slider_ranges_are_clamped(script, make_txt2img, pipeline):
    run(script, make_txt2img(), resolution=99999, tile_size=1, tile_overlap=10**6)
    assert pipeline.generation_kwargs[0]["resolution"] == 3840
    runner = pipeline.runner_kwargs[0]
    assert runner["encode_tile_size"] == runner["decode_tile_size"] == (512, 512)
    assert runner["encode_tile_overlap"] == runner["decode_tile_overlap"] == (256, 256)


def test_chosen_tile_size_and_overlap_reach_both_vae_phases(script, make_txt2img, pipeline):
    run(script, make_txt2img(), tile_size=1024, tile_overlap=256)
    runner = pipeline.runner_kwargs[0]
    assert runner["encode_tile_size"] == runner["decode_tile_size"] == (1024, 1024)
    assert runner["encode_tile_overlap"] == runner["decode_tile_overlap"] == (256, 256)


def test_run_accepts_the_arguments_of_an_older_ui(script, make_txt2img):
    """API callers built for the version before After run / BlockSwap pass 13 arguments."""
    args = list(run_args().values())[:13]
    processed = script.run(make_txt2img(), *args)
    assert processed.images[0].size == UPSCALED_SIZE


def test_run_with_the_arguments_before_the_size_modes_keeps_the_shortest_edge(script, pipeline):
    """API callers built before the size modes pass 16 arguments, ending with the BlockSwap I/O flag."""
    args = list(run_args(resolution=1536).values())[:16]
    processed = script.run(make_img2img([Image.new("RGB", (512, 384)), Image.new("RGB", (1024, 768))]), *args)
    assert [kwargs["resolution"] for kwargs in pipeline.generation_kwargs] == [1536, 1536]
    assert all("SeedVR2 Resolution: 1536" in infotext and "SeedVR2 Size Mode: Shortest edge" in infotext
               for infotext in processed.infotexts)


# --- output size modes -----------------------------------------------------------------------------------------------

def test_scale_factor_gives_each_input_size_its_own_resolution(script, pipeline, capsys):
    inputs = [Image.new("RGB", (512, 384)), Image.new("RGB", (1024, 768)), Image.new("RGB", (512, 384))]
    processed = run(script, make_img2img(inputs), size_mode="Scale factor", scale_factor=2)
    assert [(len(kwargs_pass), kwargs["resolution"], kwargs["max_resolution"])
            for kwargs_pass, kwargs in zip(pipeline.passes, pipeline.generation_kwargs)] == [(2, 768, 3072),
                                                                                            (1, 1536, 3840)]
    assert ["SeedVR2 Resolution: 768" in processed.infotexts[0], "SeedVR2 Resolution: 1536" in processed.infotexts[1],
            "SeedVR2 Resolution: 768" in processed.infotexts[2]] == [True] * 3
    assert all("SeedVR2 Size Mode: Scale factor, SeedVR2 Scale: 2.0" in infotext for infotext in processed.infotexts)
    console = capsys.readouterr().out
    assert "2 images 512x384 -> 1024x768" in console and "1 image 1024x768 -> 2048x1536" in console


def test_long_edge_mode_names_the_long_edge_and_keeps_it(script, pipeline):
    processed = run(script, make_img2img([Image.new("RGB", (1920, 1080))]), size_mode="Long edge", long_edge=2048)
    assert pipeline.generation_kwargs[0]["resolution"] == 1152
    assert "SeedVR2 Resolution: 1152" in processed.infotexts[0]
    assert "SeedVR2 Size Mode: Long edge, SeedVR2 Long Edge: 2048" in processed.infotexts[0]


def test_small_input_raised_to_512_is_noted_in_the_console(script, pipeline, capsys):
    run(script, make_img2img([Image.new("RGB", (384, 384))]), size_mode="Scale factor", scale_factor=1)
    assert pipeline.generation_kwargs[0]["resolution"] == 512
    assert "384x384 -> 512x512 (raised to the 512 px minimum)" in capsys.readouterr().out


@pytest.mark.parametrize("size_mode, scale_factor, long_edge", [("8K", 2, 2048), (None, "x", None), (3, -1, 10**9),
                                                                 ("Scale factor", float("nan"), "abc"),
                                                                 ("Long edge", None, float("inf"))])
def test_hostile_size_values_from_the_api_fall_back_instead_of_failing(script, pipeline, size_mode, scale_factor,
                                                                       long_edge):
    processed = run(script, make_img2img([Image.new("RGB", (1024, 576))]), size_mode=size_mode,
                    scale_factor=scale_factor, long_edge=long_edge)
    assert "SeedVR2 Error" not in processed.infotexts[0]
    assert 512 <= pipeline.generation_kwargs[0]["resolution"] <= 3840


def test_size_mode_shows_only_its_own_control(script):
    script.ui(is_img2img=False)
    assert [update["visible"] for update in script._show_size_controls("Scale factor")] == [False, True, False]
    assert [update["visible"] for update in script._show_size_controls("Long edge")] == [False, False, True]
    assert [update["visible"] for update in script._show_size_controls("garbage")] == [True, False, False]


def test_size_controls_start_in_the_shortest_edge_mode_with_the_old_label(script):
    """The old label keeps the px value users saved in ui-config.json; new controls have new labels."""
    components = {component.kwargs["label"]: component for component in script.ui(is_img2img=True)}
    assert components["Output Size Mode"].kwargs["value"] == "Shortest edge"
    assert components["Upscale Resolution (Shortest Edge)"].kwargs["value"] == 1080
    assert (components["Scale Factor"].kwargs["value"], components["Output Long Edge (px)"].kwargs["value"]) == (2.0,
                                                                                                                2048)


def test_quick_scale_buttons_set_the_factor(script, script_module, monkeypatch):
    buttons = []
    button_class = script_module.gr.Button

    def recording_button(*args, **kwargs):
        buttons.append(button_class(*args, **kwargs))
        return buttons[-1]

    monkeypatch.setattr(script_module.gr, "Button", recording_button)
    components = {component.kwargs["label"]: component for component in script.ui(is_img2img=True)}
    quick = [button for button in buttons if str(button.kwargs.get("value", "")).startswith("×")]
    assert {button.kwargs["value"]: button.events[0]["fn"]() for button in quick} == {"×1.5": 1.5, "×2": 2.0,
                                                                                      "×4": 4.0}
    assert all(button.events[0]["outputs"] == [components["Scale Factor"]] for button in quick)


# --- the calls into the vendored pipeline ----------------------------------------------------------------------------

def real_parameters(module_name, function_name):
    """Positional parameter names, keyword names, required names and whether `**kwargs` is accepted, read from the
    vendored source (not imported: it needs the pipeline's packages)."""
    path = os.path.join(REPO_ROOT, *module_name.split(".")) + ".py"
    with open(path, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    arguments = function.args
    positional = [arg.arg for arg in arguments.posonlyargs + arguments.args]
    keywords = positional + [arg.arg for arg in arguments.kwonlyargs]
    required = positional[:len(positional) - len(arguments.defaults)] + [
        arg.arg for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults) if default is None]
    return positional, keywords, required, arguments.kwarg is not None


def test_every_pipeline_call_fits_the_real_vendored_signature(script, make_txt2img, pipeline):
    """The fake pipeline accepts anything; a keyword renamed upstream must still fail a test, not every user's run."""
    run(script, make_txt2img(batch_size=2))
    called = {(call.module, call.function) for call in pipeline.calls}
    assert called == {(module, name) for module, names in PIPELINE_FUNCTIONS.items() for name in names}
    for call in pipeline.calls:
        positional, keywords, required, takes_any_keyword = real_parameters(call.module, call.function)
        assert call.positional_count <= len(positional), call
        supplied = set(positional[:call.positional_count]) | set(call.keywords)
        assert set(required) <= supplied, (call, set(required) - supplied)
        assert takes_any_keyword or set(call.keywords) <= set(keywords), (call, set(call.keywords) - set(keywords))
        assert not set(call.keywords) & set(positional[:call.positional_count]), call


# --- output size preview ---------------------------------------------------------------------------------------------

def forge_txt2img_size_controls(script, gr, width=512, height=384, enable_hr=False, hr_scale=2.0):
    """Forge's txt2img size controls as Forge builds them before the scripts' UI, passed to `after_component`."""
    controls = {
        "txt2img_width": gr.Slider(value=width, elem_id="txt2img_width"),
        "txt2img_height": gr.Slider(value=height, elem_id="txt2img_height"),
        "txt2img_hr-checkbox": gr.Checkbox(value=enable_hr, elem_id="txt2img_hr-checkbox"),
        "txt2img_hr_scale": gr.Slider(value=hr_scale, elem_id="txt2img_hr_scale"),
        "txt2img_hr_resize_x": gr.Slider(value=0, elem_id="txt2img_hr_resize_x"),
        "txt2img_hr_resize_y": gr.Slider(value=0, elem_id="txt2img_hr_resize_y"),
    }
    script.after_component(gr.Accordion(elem_id="txt2img_hr"), elem_id="txt2img_hr")  # the accordion is not the flag
    for elem_id, component in controls.items():
        script.after_component(component, **component.kwargs)
    return controls


def output_size_note(script, is_img2img):
    """The Markdown the preview writes to: the first Markdown created after the size controls."""
    created = []
    gr = sys.modules["gradio"]
    markdown_class = gr.Markdown

    def recording_markdown(*args, **kwargs):
        created.append(markdown_class(*args, **kwargs))
        return created[-1]

    gr.Markdown = recording_markdown
    try:
        components = script.ui(is_img2img=is_img2img)
    finally:
        gr.Markdown = markdown_class
    note = next(markdown for markdown in created if str(markdown.value).startswith("Output size"))
    return note, {component.kwargs["label"]: component for component in components}


def test_txt2img_preview_starts_with_the_size_of_forge_s_controls(script):
    forge_txt2img_size_controls(script, sys.modules["gradio"], width=832, height=1216)
    note, _ = output_size_note(script, is_img2img=False)
    assert note.value == "Output size: **1080×1578** from 832×1216"


def test_txt2img_preview_follows_every_size_control(script):
    forge = forge_txt2img_size_controls(script, sys.modules["gradio"])
    note, ours = output_size_note(script, is_img2img=False)
    watched = list(forge.values()) + [ours["Upscale Resolution (Shortest Edge)"], ours["Output Size Mode"],
                                      ours["Scale Factor"], ours["Output Long Edge (px)"]]
    for component in watched:
        event = next(event for event in component.events if event["outputs"] == [note])
        assert event["inputs"] == watched
    handler = next(event for event in forge["txt2img_width"].events if event["outputs"] == [note])["fn"]
    # 512x384 with Hires. fix x2 -> 1024x768, then the scale factor x2
    assert handler(512, 384, True, 2.0, 0, 0, 1080, "Scale factor", 2.0, 2048) == \
        "Output size: **2048×1536** from 1024×768"


def test_txt2img_preview_without_forge_s_controls_stays_generic(script):
    note, ours = output_size_note(script, is_img2img=False)
    assert note.value == "Output size: depends on each input image"
    assert not any(event["outputs"] == [note] for event in ours["Scale Factor"].events)


def test_img2img_preview_reads_the_image_size_in_the_browser(script):
    note, ours = output_size_note(script, is_img2img=True)
    event = next(event for event in ours["Scale Factor"].events if event["outputs"] == [note])
    assert event["_js"] == "seedvr2Img2imgPreview"
    assert event["inputs"][2:] == [ours["Upscale Resolution (Shortest Edge)"], ours["Output Size Mode"],
                                   ours["Scale Factor"], ours["Output Long Edge (px)"]]
    assert event["fn"](512, 384, 1080, "Scale factor", 1.5, 2048) == "Output size: **768×576** from 512×384"
    assert event["fn"](0, 0, 1080, "Scale factor", 1.5, 2048) == "Output size: depends on each input image"


def test_img2img_refresh_button_id_matches_the_script_s_javascript(script_module):
    with open(os.path.join(REPO_ROOT, "javascript", "seedvr2.js"), encoding="utf-8") as js:
        source = js.read()
    assert f'"{script_module.IMG2IMG_SIZE_REFRESH_ID}"' in source
    assert "function seedvr2Img2imgPreview" in source

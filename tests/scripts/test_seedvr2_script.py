"""The Forge script `scripts/seedvr2_script.py`, run through `Script.run()` against the stubs of `forge_stubs.py`.

These tests describe what users see: which images come back, with which infotext, what is saved, what happens to the
SD checkpoint and the SeedVR2 models, and how errors and interrupts end a run.
"""

import inspect
import sys

import pytest
from PIL import Image

from conftest import DIT, VAE, make_img2img, run_args
from forge_stubs import UPSCALE_FACTOR, base_image_color

UPSCALED_SIZE = (8 * UPSCALE_FACTOR, 8 * UPSCALE_FACTOR)
CACHE_IDS = ["forge_dit", "forge_vae"]


def run(script, p, **overrides):
    return script.run(p, **run_args(**overrides))


def corner_color(image):
    return image.convert("RGB").getpixel((0, 0))


def colors_match(actual, expected, tolerance=2):
    """The pipeline works in float16 and the script truncates to 8 bits, so a channel may drift by a level or two."""
    return len(actual) == len(expected) and all(
        abs(a - e) <= tolerance for actual_color, expected_color in zip(actual, expected)
        for a, e in zip(actual_color, expected_color))


# --- UI --------------------------------------------------------------------------------------------------------------

def test_ui_returns_one_component_per_run_argument(script):
    components = script.ui(is_img2img=False)
    run_parameters = list(inspect.signature(script.run).parameters)[1:]  # without `p`
    assert len(components) == len(run_parameters)
    assert list(run_args()) == run_parameters


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


def test_a_failing_save_keeps_the_result(script, make_txt2img, forge):
    forge.save_error = OSError("disk full")
    processed = run(script, make_txt2img(batch_size=2))
    assert [image.size for image in processed.images] == [UPSCALED_SIZE] * 2


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
    assert pipeline.cache.cached == set()
    assert pipeline.runner_kwargs[0]["dit_cache"] is False


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
    assert runner["encode_tile_size"] == (512, 512) and runner["encode_tile_overlap"] == (512, 512)


def test_run_accepts_the_arguments_of_an_older_ui(script, make_txt2img):
    """API callers built for the version before After run / BlockSwap pass 13 arguments."""
    args = list(run_args().values())[:13]
    processed = script.run(make_txt2img(), *args)
    assert processed.images[0].size == UPSCALED_SIZE

"""The Forge script's run feedback and settings: the title, Forge's progress bar, color correction, the attention
backend and the GPU profiles (`scripts/seedvr2_script.py`), run against the stubs of `forge_stubs.py`."""

import pytest
from PIL import Image

from forge_stubs import DIT, make_img2img, run_args


def components_by_label(script, is_img2img=False):
    return {component.kwargs["label"]: component for component in script.ui(is_img2img=is_img2img)}


def gpu_profile_wiring(script):
    """The GPU profile dropdown (not a run argument) and its event, found through the DiT dropdown's events."""
    components = components_by_label(script)
    event = next(event for event in components["DiT Model"].events if event["fn"].__name__ == "apply_gpu_profile")
    return components, event["inputs"][0], event


# --- title -----------------------------------------------------------------------------------------------------------

def test_the_title_has_no_vram_class(script):
    assert script.title() == "SeedVR2 Upscaler"


# --- Forge's progress bar --------------------------------------------------------------------------------------------

def test_txt2img_upscale_is_one_job_whose_bar_runs_from_start_to_end(script, make_txt2img, forge, pipeline):
    forge.state.job_count, forge.state.job_no = 4, 4  # what Forge leaves after generating four images
    script.run(make_txt2img(n_iter=4), **run_args())
    assert (forge.state.job_count, forge.state.job_no) == (1, 0)
    assert pipeline.progress_seen == sorted(pipeline.progress_seen)
    assert 0 < pipeline.progress_seen[0] < 1
    assert forge.state.progress_fraction() == pytest.approx(1.0)
    assert forge.state.textinfo.startswith("SeedVR2: ")


def test_a_single_img2img_run_gets_a_job_of_its_own(script, forge, pipeline):
    forge.state.job_count = -1  # what `state.begin()` sets
    script.run(make_img2img([Image.new("RGB", (8, 8))]), **run_args())
    assert forge.state.job_count == 1
    assert forge.state.progress_fraction() == pytest.approx(1.0)


def test_img2img_batch_keeps_forge_s_job_counter_and_moves_past_each_file(script, forge, pipeline):
    """In img2img Batch Forge sets `job_count` to files × n_iter once and leaves `job_no` to `process_images`, which
    the script replaces (`modules/img2img.py`): the upscale moves the current file's share, then past the file."""
    forge.state.job_count, forge.state.job_no = 3, 0  # what Forge sets before the first of three files
    for file_number in range(3):
        pipeline.progress_seen.clear()
        script.run(make_img2img([Image.new("RGB", (8, 8))]), **run_args())
        assert all(file_number / 3 <= fraction <= (file_number + 1) / 3 for fraction in pipeline.progress_seen)
        assert (forge.state.job_count, forge.state.job_no) == (3, file_number + 1)
    assert forge.state.progress_fraction() == pytest.approx(1.0)


def test_img2img_batch_with_several_iterations_moves_past_all_of_them(script, forge, pipeline):
    """Forge counts `n_iter` jobs per file; the script upscales the file once and moves past all of its jobs."""
    forge.state.job_count, forge.state.job_no = 4, 0  # two files, n_iter 2
    script.run(make_img2img([Image.new("RGB", (8, 8))], n_iter=2), **run_args())
    assert forge.state.job_no == 2


@pytest.mark.parametrize("job_count, job_no", [(None, None), ("x", "3"), (2, 5), (0, 0), (-1, 0)])
def test_odd_job_counters_make_the_upscale_its_own_job(script, forge, pipeline, job_count, job_no):
    forge.state.job_count, forge.state.job_no = job_count, job_no
    script.run(make_img2img([Image.new("RGB", (8, 8))]), **run_args())
    assert (forge.state.job_count, forge.state.job_no) == (1, 0)
    assert forge.state.progress_fraction() == pytest.approx(1.0)


def test_several_passes_move_the_bar_forward_only(script, forge, pipeline):
    inputs = [Image.new("RGB", (8, 8)), Image.new("RGB", (12, 8)), Image.new("RGB", (8, 12))]
    script.run(make_img2img(inputs), **run_args())
    assert len(pipeline.passes) == 3
    assert pipeline.progress_seen == sorted(pipeline.progress_seen)
    assert "pass 3 of 3" in forge.state.textinfo


# --- color correction and attention -----------------------------------------------------------------------------------

def test_the_ui_offers_every_color_correction_with_lab_first(script):
    dropdown = components_by_label(script)["Color Correction"]
    assert dropdown.kwargs["choices"] == ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]
    assert dropdown.kwargs["value"] == "lab"


def test_the_ui_offers_sdpa_and_only_installed_backends(script_module, monkeypatch):
    installed = {"sageattention"}
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *args: object() if name in installed else None)
    dropdown = components_by_label(script_module.Script())["Attention"]
    assert dropdown.kwargs["choices"] == ["sdpa", "sageattn_2"] and dropdown.kwargs["value"] == "sdpa"


def test_chosen_color_correction_and_attention_reach_the_pipeline_and_the_infotext(script, pipeline):
    pipeline.attention_available = ["SAGE_ATTN_2_AVAILABLE"]
    processed = script.run(make_img2img([Image.new("RGB", (8, 8))]),
                           **run_args(color_correction="hsv", attention_mode="sageattn_2"))
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sageattn_2"
    assert "SeedVR2 Color Correction: hsv, SeedVR2 Attention: sageattn_2" in processed.infotexts[0]


def test_an_attention_backend_that_did_not_import_falls_back_to_sdpa(script, pipeline, capsys):
    processed = script.run(make_img2img([Image.new("RGB", (8, 8))]), **run_args(attention_mode="flash_attn_2"))
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sdpa"
    assert "SeedVR2 Attention: sdpa" in processed.infotexts[0]
    assert "'flash_attn_2' is not available" in capsys.readouterr().out


@pytest.mark.parametrize("color_correction, attention_mode", [("rgb", "xformers"), (None, None), (7, ["sdpa"]),
                                                               ("", "")])
def test_unknown_api_values_fall_back_to_lab_and_sdpa(script, pipeline, color_correction, attention_mode):
    processed = script.run(make_img2img([Image.new("RGB", (8, 8))]),
                           **run_args(color_correction=color_correction, attention_mode=attention_mode))
    assert processed.images[0].size == (16, 16)
    assert "SeedVR2 Color Correction: lab, SeedVR2 Attention: sdpa" in processed.infotexts[0]


def test_a_call_from_before_the_options_gets_lab_and_sdpa(script, pipeline):
    """API callers built for the output size modes pass 19 arguments."""
    args = list(run_args().values())[:19]
    processed = script.run(make_img2img([Image.new("RGB", (8, 8))]), *args)
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sdpa"
    assert "SeedVR2 Color Correction: lab" in processed.infotexts[0]


# --- GPU profiles -----------------------------------------------------------------------------------------------------

def test_the_gpu_profile_starts_at_manual_is_not_saved_and_is_not_a_run_argument(script):
    components, gpu_profile, _ = gpu_profile_wiring(script)
    assert gpu_profile.kwargs["value"] == "Manual"
    assert gpu_profile.kwargs["choices"] == ["Manual", "8 GB", "12 GB", "24 GB"]
    assert gpu_profile.do_not_save_to_config is True
    assert "GPU Profile" not in components


def test_a_profile_fills_blockswap_and_the_vae_tiles(script, script_module):
    components, gpu_profile, event = gpu_profile_wiring(script)
    assert event["outputs"] == [components["BlockSwap: DiT blocks kept in RAM"], components["VAE Tile Size"],
                                components["Tile Overlap"]]
    assert [e["outputs"] for e in gpu_profile.events if e["event"] == "change"] == [event["outputs"]]
    blocks, tile, overlap = script_module.apply_gpu_profile("24 GB", DIT)
    assert (blocks["value"], tile["value"], overlap["value"]) == (0, 1024, 128)


def test_a_7b_dit_gets_the_profile_s_7b_blockswap(script_module, monkeypatch):
    monkeypatch.setattr(script_module, "is_7b_checkpoint", lambda name, path: True)
    assert script_module.apply_gpu_profile("8 GB", DIT)[0]["value"] == 16
    monkeypatch.setattr(script_module, "is_7b_checkpoint", lambda name, path: False)
    assert script_module.apply_gpu_profile("8 GB", DIT)[0]["value"] == 0


@pytest.mark.parametrize("label, dit", [("Manual", DIT), ("16 GB", DIT), (None, None), ("8 GB", "gone.gguf")])
def test_manual_unknown_profiles_and_missing_models_never_break_the_sliders(script_module, label, dit):
    updates = script_module.apply_gpu_profile(label, dit)
    assert len(updates) == 3
    if label == "8 GB":  # a DiT that is not listed counts as 3B
        assert updates[0]["value"] == 0
    else:
        assert all("value" not in update for update in updates)


def test_moving_a_slider_by_hand_sets_the_profile_back_to_manual(script):
    components, gpu_profile, event = gpu_profile_wiring(script)
    for slider in event["outputs"]:
        inputs = [e for e in slider.events if e["event"] == "input"]
        assert len(inputs) == 1 and inputs[0]["outputs"] == [gpu_profile] and inputs[0]["fn"]() == "Manual"

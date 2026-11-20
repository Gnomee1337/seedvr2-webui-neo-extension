"""Hostile values for the run feedback and settings of the Forge script `scripts/seedvr2_script.py` (color correction,
attention, Forge's progress bar, GPU profiles), against the stubs of `forge_stubs.py`.

API callers send `script_args` positionally and of any JSON type; another extension or the API path may leave
`shared.state` in an odd shape; a Gradio client can call the GPU profile event with any value. Checked here: no value
crashes a run or turns into an error infotext, the pipeline and the infotext only ever get a known option, nothing a
caller sends reaches the infotext as text, the bar stays within 0…1, and a profile only ever fills values the sliders
accept.
"""
import sys
from enum import Enum

import pytest
from PIL import Image

from forge_stubs import DIT, make_img2img, run_args
from seedvr2_neo.pipeline_options import AttentionMode, ColorCorrection

GARBAGE = [None, "", "   ", "rgb", "lab, Steps: 999", "hsv\nSteps: 999", "sdpa\nSteps: 999", "💥" * 5000, "x" * 200_000,
           float("nan"), float("inf"), -1, 0, 10 ** 400, True, False, [], ["hsv"], {}, {"value": "hsv"}, b"hsv",
           b"sageattn_2", object(), complex(1, 2), "ｈｓｖ", "sdpa​"]

FLAGS = {AttentionMode.FLASH_ATTN_2: "FLASH_ATTN_2_AVAILABLE", AttentionMode.FLASH_ATTN_3: "FLASH_ATTN_3_AVAILABLE",
         AttentionMode.SAGEATTN_2: "SAGE_ATTN_2_AVAILABLE", AttentionMode.SAGEATTN_3: "SAGE_ATTN_3_AVAILABLE"}


def short_id(value):
    return repr(value)[:30]


def one_input():
    return make_img2img([Image.new("RGB", (8, 8))])


def assert_clean(processed, color="lab", attention="sdpa"):
    infotext = "".join(processed.infotexts)
    assert processed.images[0].size == (16, 16)
    assert "SeedVR2 Error" not in infotext
    assert f"SeedVR2 Color Correction: {color}, SeedVR2 Attention: {attention}" in infotext
    assert infotext.count("SeedVR2 Color Correction: ") == infotext.count("SeedVR2 Attention: ") == 1
    assert "Steps: 999" not in infotext


# --- color correction and attention from the API ------------------------------------------------------------------

@pytest.mark.parametrize("argument", ["color_correction", "attention_mode"])
@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_in_one_option_falls_back_and_never_reaches_the_infotext(script, pipeline, argument, value):
    pipeline.attention_available = list(FLAGS.values())  # even with every backend imported
    processed = script.run(one_input(), **run_args(**{argument: value, "seed": 42}))
    assert_clean(processed)
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sdpa"


@pytest.mark.parametrize("member", list(AttentionMode))
def test_an_attention_member_sent_as_color_correction_is_lab(script, pipeline, member):
    assert_clean(script.run(one_input(), **run_args(color_correction=member, attention_mode=ColorCorrection.HSV)))
    assert pipeline.runner_kwargs[0]["attention_mode"] == "sdpa"


@pytest.mark.parametrize("member", list(AttentionMode))
def test_every_backend_passes_when_imported_and_falls_back_when_not(script_module, forge, pipeline, member, capsys):
    pipeline.attention_available = list(FLAGS.values())
    processed = script_module.Script().run(one_input(), **run_args(attention_mode=f"  {member.value.upper()} "))
    assert_clean(processed, attention=member.value)
    pipeline.attention_available = []
    processed = script_module.Script().run(one_input(), **run_args(attention_mode=member.value))
    assert_clean(processed)
    assert [kwargs["attention_mode"] for kwargs in pipeline.runner_kwargs] == [member.value, "sdpa"]
    note = "is not available" in capsys.readouterr().out
    assert note == (member is not AttentionMode.SDPA)


@pytest.mark.parametrize("member", list(ColorCorrection))
def test_every_color_correction_reaches_the_infotext_in_any_case(script, pipeline, member):
    assert_clean(script.run(one_input(), **run_args(color_correction=member.value.title())), color=member.value)


def test_a_flag_list_with_junk_names_picks_nothing_extra(script, pipeline):
    pipeline.attention_available = ["SDPA_AVAILABLE", "XFORMERS_AVAILABLE", "sageattn_2", ""]
    assert_clean(script.run(one_input(), **run_args(attention_mode="sageattn_2")))


@pytest.mark.parametrize("count", [20, 21])
def test_calls_with_one_or_both_new_options_read_positionally(script, pipeline, count):
    pipeline.attention_available = ["SAGE_ATTN_2_AVAILABLE"]
    args = list(run_args(color_correction="adain", attention_mode="sageattn_2").values())[:count]
    processed = script.run(one_input(), *args)
    assert_clean(processed, color="adain", attention="sageattn_2" if count == 21 else "sdpa")


def test_a_foreign_str_enum_counts_by_its_value(script, pipeline):
    class Remote(str, Enum):
        COLOR = "wavelet"

    assert_clean(script.run(one_input(), **run_args(color_correction=Remote.COLOR)), color="wavelet")


# --- Forge's progress bar with odd state -------------------------------------------------------------------------------

@pytest.mark.parametrize("job_count, job_no", [(-1, 0), (0, 0), (0, 7), (1, 0), (1, 1), (2, 5), (5, 4), (10 ** 9, 0),
                                               (3, 3), (2.0, 1.0), (True, False)])
def test_odd_job_counters_never_fail_the_run_and_keep_the_bar_in_0_to_1(script, forge, pipeline, job_count, job_no):
    forge.state.job_count, forge.state.job_no = job_count, job_no
    forge.state.sampling_steps, forge.state.sampling_step = 37, 999  # left by the sampler of the base image
    processed = script.run(one_input(), **run_args())
    assert "SeedVR2 Error" not in processed.infotexts[0]
    assert pipeline.progress_seen and all(0.0 <= fraction <= 1.0 for fraction in pipeline.progress_seen)
    assert pipeline.progress_seen == sorted(pipeline.progress_seen)
    assert forge.state.sampling_steps == 1000 and 0 <= forge.state.sampling_step <= 1000
    assert forge.state.job_count >= 1 and 0 <= forge.state.job_no <= forge.state.job_count
    assert 0.0 <= forge.state.progress_fraction() <= 1.0


def test_a_negative_job_number_is_left_alone_and_does_not_fail_the_run(script, forge, pipeline):
    """Forge never sets it; the script must not crash on it (the bar it shows is Forge's formula)."""
    forge.state.job_count, forge.state.job_no = 4, -2
    assert "SeedVR2 Error" not in script.run(one_input(), **run_args()).infotexts[0]


@pytest.mark.parametrize("job_count, job_no", [(None, 0), ("3", 0), (2, None), (2, "1"), (float("nan"), 0),
                                               (float("inf"), 0), (object(), []), (10 ** 400, 0)], ids=short_id)
def test_non_numeric_job_counters_never_fail_the_run(script, forge, pipeline, monkeypatch, job_count, job_no):
    """Forge never sets these; the script reads them leniently. Values it leaves as Forge's own would break Forge's
    progress endpoint (another thread in Forge), not the run, so the stub's bar reading is switched off here."""
    monkeypatch.setattr(forge.state, "progress_fraction", lambda: 0.0)
    forge.state.job_count, forge.state.job_no = job_count, job_no
    processed = script.run(one_input(), **run_args())
    assert "SeedVR2 Error" not in processed.infotexts[0]


@pytest.mark.parametrize("fraction, step", [(float("inf"), 1000), (float("-inf"), 0), (-5, 0), (7, 1000), (0.0004, 0),
                                            (0.0005, 0), (0.0006, 1), (0.9996, 1000), (True, 1000)])
def test_the_bar_is_clamped_to_its_steps(script_module, forge, fraction, step):
    script_module.report_forge_progress(fraction, "SeedVR2: Encoding")
    assert forge.state.sampling_step == step and forge.state.textinfo == "SeedVR2: Encoding"


def test_a_nan_from_the_pipeline_callback_never_fails_the_upscale(script, forge, pipeline, monkeypatch):
    phases = sys.modules["seedvr2_vendor.core.generation_phases"]
    original = phases.upscale_all_batches

    def upscale(runner, ctx, progress_callback=None, **kwargs):
        progress_callback(float("nan"), 1)
        return original(runner, ctx, progress_callback=progress_callback, **kwargs)

    monkeypatch.setattr(phases, "upscale_all_batches", upscale)
    processed = script.run(one_input(), **run_args())
    assert "SeedVR2 Error" not in processed.infotexts[0]


def test_an_interrupt_mid_run_leaves_the_bar_inside_0_to_1(script, forge, pipeline):
    pipeline.interrupt_in = "decode"
    processed = script.run(one_input(), **run_args())
    assert "Interrupted" in processed.infotexts[0]
    assert 0.0 <= forge.state.progress_fraction() <= 1.0 and forge.state.textinfo.startswith("SeedVR2: ")


# --- GPU profiles ------------------------------------------------------------------------------------------------------

LABELS = [None, "", "Manual", "manual", "8 gb", " 8 GB", "8 GB\n", b"8 GB", ["8 GB"], {"8 GB": 1}, 8, float("nan"),
          True, object(), "💥" * 5000, "8 GB" * 50_000, "16 GB"]
DIT_NAMES = [None, "", DIT, DIT.upper(), "../" + DIT, "..\\..\\" + DIT, "gone.gguf", "x" * 100_000, "\x00", "💥",
             [DIT], {DIT: 1}, 5, b"dit.gguf", object()]


@pytest.mark.parametrize("label", LABELS, ids=short_id)
def test_junk_profile_labels_leave_every_slider_alone(script_module, label):
    updates = script_module.apply_gpu_profile(label, DIT)
    assert len(updates) == 3 and all("value" not in update for update in updates)


@pytest.mark.parametrize("dit", DIT_NAMES, ids=short_id)
@pytest.mark.parametrize("label", ["8 GB", "12 GB", "24 GB"])
def test_junk_dit_names_get_the_3b_values_without_reading_any_file(script_module, monkeypatch, label, dit):
    """Anything that is not a listed DiT counts as 3B; the size check never runs for a path the user did not pick."""
    checked = []
    monkeypatch.setattr(script_module, "is_7b_checkpoint", lambda name, path: checked.append(path) or True)
    blocks, tile, overlap = script_module.apply_gpu_profile(label, dit)
    profile = script_module.find_profile(label)
    expected_7b = dit == DIT
    assert blocks["value"] == profile.blocks_to_swap(expected_7b)
    assert (tile["value"], overlap["value"]) == (profile.tile_size, profile.tile_overlap)
    assert bool(checked) == expected_7b


def test_a_profile_with_the_model_folder_gone_still_fills_the_3b_values(script_module, forge):
    import os
    import shutil
    shutil.rmtree(os.path.join(forge.models_path, "SeedVR2"))
    blocks, tile, _ = script_module.apply_gpu_profile("24 GB", DIT)
    assert (blocks["value"], tile["value"]) == (0, 1024)


@pytest.mark.parametrize("label", ["8 GB", "12 GB", "24 GB"])
@pytest.mark.parametrize("is_7b", [False, True])
def test_every_profile_value_fits_the_built_slider(script, script_module, monkeypatch, label, is_7b):
    monkeypatch.setattr(script_module, "is_7b_checkpoint", lambda name, path: is_7b)
    components = {component.kwargs["label"]: component for component in script.ui(is_img2img=False)}
    sliders = [components["BlockSwap: DiT blocks kept in RAM"], components["VAE Tile Size"], components["Tile Overlap"]]
    for slider, update in zip(sliders, script_module.apply_gpu_profile(label, DIT)):
        value, kwargs = update["value"], slider.kwargs
        assert kwargs["minimum"] <= value <= kwargs["maximum"], (slider.kwargs["label"], value)
        steps = (value - kwargs["minimum"]) / kwargs["step"]
        assert abs(steps - round(steps)) < 1e-9, (slider.kwargs["label"], value)


def test_a_profile_s_values_run_unchanged(script, script_module, pipeline, monkeypatch):
    """What a profile fills is what the run uses: the tile values are not clamped or snapped by the run."""
    monkeypatch.setattr(script_module, "is_7b_checkpoint", lambda name, path: True)
    blocks, tile, overlap = (update["value"] for update in script_module.apply_gpu_profile("8 GB", DIT))
    script.run(one_input(), **run_args(blocks_to_swap=blocks, tile_size=tile, tile_overlap=overlap))
    kwargs = pipeline.runner_kwargs[0]
    assert kwargs["decode_tile_size"] == (tile, tile) and kwargs["decode_tile_overlap"] == (overlap, overlap)
    assert kwargs["block_swap_config"]["blocks_to_swap"] == blocks


@pytest.mark.parametrize("n_iter", [None, "abc", "2", -5, 0, 1, 7, 10 ** 400, float("nan"), float("inf"), [], True])
def test_img2img_batch_with_junk_iterations_moves_the_job_counter_inside_its_range(script, forge, pipeline, n_iter):
    """In img2img Batch the run moves Forge's counter past the file by `n_iter`; a junk count never fails the run or
    pushes the counter past `job_count`."""
    forge.state.job_count, forge.state.job_no = 3, 1
    p = make_img2img([Image.new("RGB", (8, 8))])
    p.n_iter = n_iter
    processed = script.run(p, **run_args())
    assert "SeedVR2 Error" not in processed.infotexts[0]
    assert forge.state.job_count == 3 and 2 <= forge.state.job_no <= 3
    assert 0.0 <= forge.state.progress_fraction() <= 1.0

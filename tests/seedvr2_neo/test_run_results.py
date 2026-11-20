"""Saving rules, generation parameters and the result list of a run — `seedvr2_neo/run_results.py`."""
import pytest

from seedvr2_neo import run_results as rr
from seedvr2_neo.upscale_settings import parse_upscale_settings


def settings(**overrides):
    values = dict(resolution=1080, input_noise=0.0, latent_noise=0.0, use_tile_vae=True, tile_size=512,
                  tile_overlap=128)
    values.update(overrides)
    return parse_upscale_settings(**values)


# --- plan_saving ---

def test_txt2img_with_saving_on_saves_results_and_no_second_original():
    # Forge already saved the base images by the same rule
    plan = rr.plan_saving(is_txt2img=True, samples_save=True, saving_disabled_by_caller=False,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=True, save_originals=False)


@pytest.mark.parametrize("is_txt2img", [True, False])
def test_always_save_off_writes_nothing_even_with_original_requested(is_txt2img):
    plan = rr.plan_saving(is_txt2img=is_txt2img, samples_save=False, saving_disabled_by_caller=False,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=False, save_originals=False)


def test_img2img_input_is_saved_as_original_when_requested():
    # Forge never saves the img2img input itself
    plan = rr.plan_saving(is_txt2img=False, samples_save=True, saving_disabled_by_caller=False,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=True, save_originals=True)


@pytest.mark.parametrize("is_txt2img", [True, False])
def test_api_call_without_saving_writes_nothing(is_txt2img):
    plan = rr.plan_saving(is_txt2img=is_txt2img, samples_save=True, saving_disabled_by_caller=True,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=False, save_originals=False)


def test_original_not_requested_is_never_saved():
    plan = rr.plan_saving(is_txt2img=False, samples_save=True, saving_disabled_by_caller=False,
                          save_original_requested=False)

    assert plan.save_originals is False


# --- read_option ---

def test_override_wins_over_the_user_setting():
    # the img2img Batch tab turns date subfolders off through override_settings
    assert rr.read_option({"save_to_dirs": False}, "save_to_dirs", True) is False


def test_without_override_the_default_is_used():
    assert rr.read_option({"other": 1}, "save_to_dirs", True) is True


@pytest.mark.parametrize("overrides", [None, [], "save_to_dirs", 5])
def test_missing_or_odd_override_settings_give_the_default(overrides):
    assert rr.read_option(overrides, "save_to_dirs", "default") == "default"


# --- generation parameters ---

def test_success_params_name_models_resolution_seed_and_size_mode():
    params = rr.seedvr2_generation_params("dit.safetensors", "vae.safetensors", settings(), (512, 384), 42)

    assert params == {"SeedVR2 Model": "dit.safetensors", "SeedVR2 VAE": "vae.safetensors",
                      "SeedVR2 Resolution": 1080, "SeedVR2 Seed": 42, "SeedVR2 Size Mode": "Shortest edge",
                      "SeedVR2 Color Correction": "lab", "SeedVR2 Attention": "sdpa"}


def test_scale_factor_params_name_the_factor_and_the_resolution_of_this_input():
    chosen = settings(size_mode="Scale factor", scale_factor=1.5)

    small = rr.seedvr2_generation_params("dit", "vae", chosen, (512, 384), 42)
    large = rr.seedvr2_generation_params("dit", "vae", chosen, (1024, 768), 42)

    assert (small["SeedVR2 Resolution"], large["SeedVR2 Resolution"]) == (576, 1152)
    assert small["SeedVR2 Size Mode"] == "Scale factor" and small["SeedVR2 Scale"] == 1.5
    assert "SeedVR2 Long Edge" not in small


def test_long_edge_params_name_the_long_edge():
    params = rr.seedvr2_generation_params("dit", "vae", settings(size_mode="Long edge", long_edge=2048), (1920, 1080), 1)

    assert params["SeedVR2 Size Mode"] == "Long edge" and params["SeedVR2 Long Edge"] == 2048
    assert params["SeedVR2 Resolution"] == 1152 and "SeedVR2 Scale" not in params


def test_failure_params_keep_the_attempt_and_add_the_message():
    # Forge parses the last line as parameters only with 3+ of them, so the SeedVR2 keys stay
    params = rr.seedvr2_generation_params("dit", "vae", settings(), (512, 512), 42)

    failed = rr.failure_generation_params(params, "Interrupted")

    assert failed == {**params, "SeedVR2 Error": "Interrupted"}
    assert params == rr.seedvr2_generation_params("dit", "vae", settings(), (512, 512), 42)  # not mutated


# --- BaseInfotext ---

def test_forge_infotext_gets_params_on_its_parameter_line():
    base = rr.BaseInfotext.from_forge("a cat\nNegative prompt: dog\nSteps: 20, Seed: 1")

    assert base.with_params({"SeedVR2 Seed": 42}) == "a cat\nNegative prompt: dog\nSteps: 20, Seed: 1, SeedVR2 Seed: 42"


def test_forge_infotext_without_prompt_is_only_a_parameter_line():
    # Forge strips the empty prompt, leaving just the parameters
    assert rr.BaseInfotext.from_forge("Steps: 20").with_params({"SeedVR2 Seed": 42}) == "Steps: 20, SeedVR2 Seed: 42"


def test_img2img_prompt_gets_a_new_parameter_line():
    base = rr.BaseInfotext.from_prompt("a cat", "dog")

    assert base.with_params({"SeedVR2 Seed": 42}) == "a cat\nNegative prompt: dog\nSeedVR2 Seed: 42"


def test_img2img_without_negative_prompt_has_no_negative_line():
    assert rr.BaseInfotext.from_prompt("a cat", "").with_params({"K": 1}) == "a cat\nK: 1"


def test_img2img_without_any_prompt_is_only_the_parameter_line():
    assert rr.BaseInfotext.from_prompt("", "").with_params({"K": 1}) == "K: 1"


def test_multiline_prompt_keeps_its_lines():
    base = rr.BaseInfotext.from_forge("line one\nline two\nSteps: 20")

    assert base.with_params({"K": 1}) == "line one\nline two\nSteps: 20, K: 1"


def test_no_params_leave_the_infotext_unchanged():
    assert rr.BaseInfotext.from_forge("a cat\nSteps: 20").with_params({}) == "a cat\nSteps: 20"


def test_empty_or_missing_forge_infotext_gives_only_the_params():
    assert rr.BaseInfotext.from_forge("").with_params({"K": 1}) == "K: 1"
    assert rr.BaseInfotext.from_forge(None).with_params({"K": 1}) == "K: 1"


@pytest.mark.parametrize("value", ["CUDA out of memory, tried 2 GB", "Runtime Error: x", "two\nlines", 'say "hi"'])
def test_values_that_would_break_the_line_are_quoted(value):
    formatted = rr.format_generation_params({"SeedVR2 Error": value})

    assert formatted.startswith('SeedVR2 Error: "')
    assert "\n" not in formatted


# --- merge_run_images ---

def test_every_input_gets_one_entry_in_order():
    merged = rr.merge_run_images(["a", "b", "c"], {0: "A", 2: "C"})

    assert merged == [rr.RunImage("A", True), rr.RunImage("b", False), rr.RunImage("C", True)]


def test_nothing_upscaled_returns_the_inputs():
    assert rr.merge_run_images(["a"], {}) == [rr.RunImage("a", False)]


def test_indices_outside_the_inputs_are_ignored():
    assert rr.merge_run_images(["a"], {5: "X"}) == [rr.RunImage("a", False)]


# --- describe_generated_inputs / describe_uploaded_inputs ---

def test_generated_inputs_take_seed_prompt_and_forge_infotext_by_index():
    details = rr.describe_generated_inputs([5, 6], ["a", "b"], ["a\nSteps: 20, Seed: 5"], 2)

    assert [(d.seed, d.prompt) for d in details] == [(5, "a"), (6, "b")]
    assert details[0].infotext == rr.BaseInfotext("a", "Steps: 20, Seed: 5")
    assert details[1].infotext == rr.BaseInfotext("", "")  # no infotext from Forge: only the SeedVR2 keys later


def test_uploaded_inputs_share_the_seedvr2_seed_and_the_ui_prompt():
    details = rr.describe_uploaded_inputs(42, "a dog", "blurry", 3)

    assert len(details) == 3 and {d.seed for d in details} == {42}
    assert details[0].infotext.with_params({}) == "a dog\nNegative prompt: blurry"


# --- finish_images / not_upscaled_comment / copy_keeping_saved_file ---

class FakeImage:
    def __init__(self, name):
        self.name = name
        self.info = {}

    def copy(self):
        return FakeImage(self.name + "-copy")


def _details(count):
    return rr.describe_uploaded_inputs(1, "p", "", count)


def test_finish_images_gives_one_infotext_per_input_with_the_failure_on_the_inputs_returned():
    upscaled, returned = FakeImage("up"), FakeImage("in")
    run_images = [rr.RunImage(upscaled, True), rr.RunImage(returned, False)]

    params = [{"SeedVR2 Resolution": 1080}, {"SeedVR2 Resolution": 2160}]
    results = rr.finish_images(run_images, _details(2), params, "Interrupted", embed_infotext=True)

    assert [r.image.name for r in results] == ["up", "in-copy"]  # the input itself is not changed
    assert results[0].infotext == "p\nSeedVR2 Resolution: 1080"
    assert results[1].infotext == "p\nSeedVR2 Resolution: 2160, SeedVR2 Error: Interrupted"
    assert [r.image.info[rr.PNG_INFO_KEY] for r in results] == [r.infotext for r in results]
    assert returned.info == {}


def test_finish_images_without_a_reason_says_not_upscaled_and_can_leave_the_image_info_alone():
    results = rr.finish_images([rr.RunImage(FakeImage("in"), False)], _details(1), [{}], None, embed_infotext=False)

    assert results[0].infotext.endswith("SeedVR2 Error: not upscaled")
    assert results[0].image.info == {}


def test_comment_counts_the_images_returned_without_upscaling():
    run_images = [rr.RunImage("a", True), rr.RunImage("b", False), rr.RunImage("c", False)]

    assert rr.not_upscaled_comment(run_images, "Runtime Error: OSError") == \
        "SeedVR2: Runtime Error: OSError — 2 of 3 images returned without upscaling"
    assert rr.not_upscaled_comment(run_images[:1], None) is None


def test_copy_keeps_the_file_forge_saved():
    image = FakeImage("in")
    image.already_saved_as = "outputs/1.png"

    assert rr.copy_keeping_saved_file(image).already_saved_as == "outputs/1.png"
    assert not hasattr(rr.copy_keeping_saved_file(FakeImage("x")), "already_saved_as")

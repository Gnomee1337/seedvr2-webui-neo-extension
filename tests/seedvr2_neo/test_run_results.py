"""Saving rules, generation parameters and the result list of a run — `seedvr2_neo/run_results.py`."""
import pytest

from seedvr2_neo import run_results as rr


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

def test_success_params_name_models_resolution_and_seed():
    params = rr.seedvr2_generation_params("dit.safetensors", "vae.safetensors", 1080, 42)

    assert params == {"SeedVR2 Model": "dit.safetensors", "SeedVR2 VAE": "vae.safetensors",
                      "SeedVR2 Resolution": 1080, "SeedVR2 Seed": 42}


def test_failure_params_keep_the_attempt_and_add_the_message():
    # Forge parses the last line as parameters only with 3+ of them, so the SeedVR2 keys stay
    params = rr.seedvr2_generation_params("dit", "vae", 1080, 42)

    failed = rr.failure_generation_params(params, "Interrupted")

    assert failed == {**params, "SeedVR2 Error": "Interrupted"}
    assert params == rr.seedvr2_generation_params("dit", "vae", 1080, 42)  # not mutated


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

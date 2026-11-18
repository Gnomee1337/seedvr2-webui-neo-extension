"""Saving rules, generation parameters and the result list of a run — `seedvr2_neo/run_results.py`."""
import pytest

from seedvr2_neo import run_results as rr


# --- plan_saving ---

def test_txt2img_with_saving_on_saves_results_and_no_second_original():
    # Forge already saved the base images by the same rule
    plan = rr.plan_saving(is_txt2img=True, samples_allowed=True, saving_disabled_by_caller=False,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=True, save_originals=False)


def test_txt2img_with_saving_off_writes_only_the_requested_original():
    plan = rr.plan_saving(is_txt2img=True, samples_allowed=False, saving_disabled_by_caller=False,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=False, save_originals=True)


def test_img2img_input_is_saved_as_original_when_requested():
    # Forge never saves the img2img input itself
    plan = rr.plan_saving(is_txt2img=False, samples_allowed=True, saving_disabled_by_caller=False,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=True, save_originals=True)


@pytest.mark.parametrize("is_txt2img", [True, False])
def test_api_call_without_saving_writes_nothing(is_txt2img):
    plan = rr.plan_saving(is_txt2img=is_txt2img, samples_allowed=False, saving_disabled_by_caller=True,
                          save_original_requested=True)

    assert plan == rr.SavePlan(save_results=False, save_originals=False)


def test_original_not_requested_is_never_saved():
    plan = rr.plan_saving(is_txt2img=False, samples_allowed=True, saving_disabled_by_caller=False,
                          save_original_requested=False)

    assert plan.save_originals is False


# --- generation parameters ---

def test_success_params_name_models_resolution_and_seed():
    params = rr.seedvr2_generation_params("dit.safetensors", "vae.safetensors", 1080, 42)

    assert params == {"SeedVR2 Model": "dit.safetensors", "SeedVR2 VAE": "vae.safetensors",
                      "SeedVR2 Resolution": 1080, "SeedVR2 Seed": 42}


def test_failure_param_carries_the_message():
    assert rr.failure_generation_params("Interrupted") == {"SeedVR2 Error": "Interrupted"}


def test_params_are_appended_to_the_parameter_line():
    infotext = "a cat\nNegative prompt: dog\nSteps: 20, Seed: 1"

    result = rr.append_generation_params(infotext, {"SeedVR2 Seed": 42})

    assert result == "a cat\nNegative prompt: dog\nSteps: 20, Seed: 1, SeedVR2 Seed: 42"


def test_params_without_base_infotext_stand_alone():
    assert rr.append_generation_params("", {"SeedVR2 Seed": 42}) == "SeedVR2 Seed: 42"


def test_no_params_leave_the_infotext_unchanged():
    assert rr.append_generation_params("Steps: 20", {}) == "Steps: 20"


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

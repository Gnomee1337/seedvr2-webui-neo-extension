"""Hostile values for infotexts, saving rules and result lists — `seedvr2_neo/run_results.py`.

Model names come from file names, the error text from any exception, prompts from the user or an API caller, and the base
infotext from Forge (possibly changed by other extensions). Whatever arrives, the infotext must still parse in Forge: the
last line is the parameters line (no line break inside it, at least three `key: value` matches), every key reads back
with its value, and the prompt part is untouched.

`parse_like_forge` below is a model of `parse_generation_parameters` in Forge's `modules/infotext_utils.py` (inherited
from A1111): the last line is parameters only with 3+ regex matches, quoted values are read with `json.loads`.
"""
import itertools
import json
import re
from dataclasses import FrozenInstanceError

import pytest

from seedvr2_neo import run_results as rr

# Forge / A1111 `re_param_code`
RE_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
SEEDVR2_KEYS = [param.value for param in rr.GenerationParam if param is not rr.GenerationParam.ERROR]


def parse_like_forge(infotext: str):
    """(prompt, negative prompt, params) the way Forge reads an infotext back."""
    *lines, last_line = infotext.strip().split("\n")
    if len(RE_PARAM.findall(last_line)) < 3:
        lines.append(last_line)
        last_line = ""
    prompt, negative, done_with_prompt = "", "", False
    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative += ("" if negative == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line
    params = {}
    for key, value in RE_PARAM.findall(last_line):
        if value[:1] == '"' and value[-1:] == '"':
            try:
                value = json.loads(value)
            except ValueError:
                pass
        params[key] = value
    return prompt, negative, params


def success_params(**overrides):
    params = rr.seedvr2_generation_params("seedvr2_ema_3b_fp8_e4m3fn.safetensors", "ema_vae_fp16.safetensors", 1080, 42)
    params.update(overrides)
    return params


HOSTILE_VALUES = [
    "", " ", "plain", "with, comma", "with: colon", "C:\\models\\dit.safetensors", "ends with backslash\\",
    'say "hi"', '"fully quoted"', '"', '\\"', "two\nlines", "crlf\r\nline", "\n", "trailing newline\n",
    "tab\there", "carriage\rreturn", "unicode модель 模型 🦄", "line\u2028separator", "nul\x00byte",
    "Steps: 20, Seed: 1", "Negative prompt: x", "x" * 100_000, "a, b: c\n\"d\"\\e" * 500, "  padded  ",
    "{\"json\": [1, 2]}", "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total)",
]


# --- format_generation_params: Forge must read every value back -----------------------------------------------------

@pytest.mark.parametrize("value", HOSTILE_VALUES)
def test_hostile_error_text_keeps_the_params_on_one_last_line(value):
    infotext = rr.BaseInfotext.from_forge("a cat\nSteps: 20, Seed: 1").with_params(
        rr.failure_generation_params(success_params(), value))

    assert infotext.split("\n")[0] == "a cat"
    assert infotext.count("\n") == 1
    _, _, params = parse_like_forge(infotext)
    assert params["Steps"] == "20" and params["Seed"] == "1"
    assert all(key in params for key in SEEDVR2_KEYS)


@pytest.mark.parametrize("value", [value for value in HOSTILE_VALUES if value == value.strip() and value])
def test_hostile_values_read_back_exactly(value):
    # leading/trailing spaces are excluded: Forge's regex strips them, an accepted loss
    infotext = rr.BaseInfotext.from_prompt("p", "").with_params(
        rr.failure_generation_params(success_params(), value))

    _, _, params = parse_like_forge(infotext)

    assert params["SeedVR2 Error"] == value


@pytest.mark.parametrize("model_name", [value for value in HOSTILE_VALUES if value == value.strip() and value])
def test_hostile_model_name_reads_back_and_does_not_eat_the_next_key(model_name):
    infotext = rr.BaseInfotext.from_forge("Steps: 20, Seed: 1, Size: 8x8").with_params(
        success_params(**{"SeedVR2 Model": model_name}))

    _, _, params = parse_like_forge(infotext)

    assert params["SeedVR2 Model"] == model_name
    assert params["SeedVR2 VAE"] == "ema_vae_fp16.safetensors"
    assert params["SeedVR2 Resolution"] == "1080" and params["SeedVR2 Seed"] == "42"


@pytest.mark.parametrize("value", [None, 0, -1, 2 ** 64, 1.5, float("nan"), float("inf"), True, [1, 2], (1,),
                                   {"a": 1}, b"bytes", rr.GenerationParam.SEED, object])
def test_non_string_values_are_formatted_without_raising_and_stay_on_one_line(value):
    formatted = rr.format_generation_params({"SeedVR2 Seed": value, "K2": 1, "K3": 2})

    assert "\n" not in formatted
    assert len(RE_PARAM.findall(formatted)) == 3


def test_enum_value_is_written_as_its_text():
    assert rr.format_generation_params({"K": rr.GenerationParam.SEED.value}) == "K: SeedVR2 Seed"


def test_empty_params_format_to_empty_text():
    assert rr.format_generation_params({}) == ""


def test_key_order_is_kept():
    params = {f"Key {index}": index for index in range(50)}

    assert [key for key, _ in RE_PARAM.findall(rr.format_generation_params(params))] == list(params)


def test_unicode_is_kept_readable_not_escaped():
    assert rr.format_generation_params({"K": "модель, 🦄"}) == 'K: "модель, 🦄"'


# --- BaseInfotext.from_forge ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", ["a cat", "", "multi\nline\nprompt", "a: b, c: d, e: f", "unicode 🦄 модель",
                                    "x" * 50_000, 'quote " and \\ backslash'])
def test_forge_prompt_and_negative_survive_unchanged(prompt):
    base = f"{prompt}\nNegative prompt: ugly, blurry\nSteps: 20, Sampler: Euler a, Seed: 1".strip()

    parsed_prompt, negative, params = parse_like_forge(rr.BaseInfotext.from_forge(base).with_params(success_params()))

    assert parsed_prompt == prompt
    assert negative == "ugly, blurry"
    assert params["Sampler"] == "Euler a"
    assert params["SeedVR2 Seed"] == "42"


def test_with_params_is_repeatable_and_does_not_mutate():
    base = rr.BaseInfotext.from_forge("a cat\nSteps: 20")

    first = base.with_params(success_params())
    second = base.with_params(success_params())

    assert first == second
    assert base == rr.BaseInfotext("a cat", "Steps: 20")
    with pytest.raises(FrozenInstanceError):
        base.params_line = "x"


@pytest.mark.parametrize("ending", ["\n", "\n\n", "\r\n", " \n", "\n  "])
def test_forge_infotext_ending_with_a_line_break_still_extends_forge_params_line(ending):
    infotext = rr.BaseInfotext.from_forge("a cat\nSteps: 20, Sampler: Euler, Seed: 1" + ending).with_params(
        success_params())

    prompt, _, params = parse_like_forge(infotext)

    assert prompt == "a cat"
    assert params["Seed"] == "1" and params["SeedVR2 Seed"] == "42"


@pytest.mark.parametrize("infotext", ["\n", "\n\n\n", "   ", "\r\n"])
def test_blank_forge_infotext_gives_a_params_only_infotext_that_parses(infotext):
    result = rr.BaseInfotext.from_forge(infotext).with_params(success_params())

    prompt, negative, params = parse_like_forge(result)

    assert prompt == "" and negative == ""
    assert all(key in params for key in SEEDVR2_KEYS)


def test_windows_line_endings_keep_forge_params_together_with_ours():
    infotext = rr.BaseInfotext.from_forge("a cat\r\nSteps: 20, Sampler: Euler, Seed: 1").with_params(success_params())

    prompt, _, params = parse_like_forge(infotext)

    assert prompt == "a cat"
    assert params["Seed"] == "1" and params["SeedVR2 Seed"] == "42"


def test_forge_params_line_with_trailing_comma_does_not_produce_an_empty_entry_that_breaks_parsing():
    infotext = rr.BaseInfotext.from_forge("a cat\nSteps: 20, Seed: 1,").with_params(success_params())

    _, _, params = parse_like_forge(infotext)

    assert params["Seed"] == "1" and params["SeedVR2 Model"] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def test_forge_infotext_already_holding_seedvr2_keys_reads_our_values_last():
    # another run's keys in the base line (e.g. a script that copied them): Forge keeps the last occurrence
    base = rr.BaseInfotext.from_forge("a cat\nSteps: 20, SeedVR2 Seed: 7, Seed: 1")

    _, _, params = parse_like_forge(base.with_params(success_params()))

    assert params["SeedVR2 Seed"] == "42"


def test_from_forge_on_a_non_string_does_not_raise_for_empty_values():
    for empty in (None, ""):
        assert rr.BaseInfotext.from_forge(empty) == rr.BaseInfotext("", "")


# --- BaseInfotext.from_prompt ---------------------------------------------------------------------------------------

def test_empty_prompt_with_negative_only_parses_as_negative():
    infotext = rr.BaseInfotext.from_prompt("", "ugly, blurry").with_params(success_params())

    prompt, negative, params = parse_like_forge(infotext)

    assert infotext.startswith("Negative prompt: ugly, blurry\n")
    assert (prompt, negative) == ("", "ugly, blurry")
    assert params["SeedVR2 Seed"] == "42"


@pytest.mark.parametrize("prompt, negative", [(None, None), (None, ""), ("", None), ("  ", ""), ("\n\n", "")])
def test_missing_or_blank_prompts_give_only_the_params_line(prompt, negative):
    assert rr.BaseInfotext.from_prompt(prompt, negative).with_params({"K": 1}) == "K: 1"


@pytest.mark.parametrize("prompt", ["a: 1, b: 2, c: 3", "Steps: 20, Seed: 1, Size: 8x8", "x\ny: 1, z: 2, w: 3"])
def test_prompt_that_looks_like_params_stays_the_prompt(prompt):
    infotext = rr.BaseInfotext.from_prompt(prompt, "").with_params(success_params())

    parsed_prompt, _, params = parse_like_forge(infotext)

    assert parsed_prompt == prompt
    assert "Steps" not in params and "a" not in params


@pytest.mark.parametrize("prompt", ["a cat", "multi\nline", "🦄 " * 1000, 'quote "x"', "trailing space "])
@pytest.mark.parametrize("negative", ["dog", "multi\nline negative", "Negative prompt: nested"])
def test_img2img_prompt_and_negative_read_back(prompt, negative):
    infotext = rr.BaseInfotext.from_prompt(prompt, negative).with_params(success_params())

    parsed_prompt, parsed_negative, params = parse_like_forge(infotext)

    assert parsed_prompt == prompt.strip()
    assert parsed_negative == negative
    assert all(key in params for key in SEEDVR2_KEYS)


def test_original_saved_without_params_keeps_only_the_prompt():
    # save_originals in img2img uses with_params({}); Forge must see no params line, not mistake the prompt for one
    infotext = rr.BaseInfotext.from_prompt("a cat", "dog").with_params({})

    assert infotext == "a cat\nNegative prompt: dog"
    assert parse_like_forge(infotext) == ("a cat", "dog", {})


# --- failure_generation_params ---------------------------------------------------------------------------------------

def test_failure_with_empty_attempt_still_has_the_error_key():
    assert rr.failure_generation_params({}, "Interrupted") == {"SeedVR2 Error": "Interrupted"}


def test_failure_overrides_an_error_key_already_in_the_params():
    failed = rr.failure_generation_params({"SeedVR2 Error": "old"}, "new")

    assert failed == {"SeedVR2 Error": "new"}


def test_failure_infotext_has_at_least_three_params_so_forge_reads_it():
    infotext = rr.BaseInfotext.from_prompt("a cat", "").with_params(
        rr.failure_generation_params(success_params(), "Runtime Error: boom"))

    _, _, params = parse_like_forge(infotext)

    assert params["SeedVR2 Error"] == "Runtime Error: boom"
    assert len(params) == 5


# --- seedvr2_generation_params ---------------------------------------------------------------------------------------

def test_success_params_always_reach_forge_three_param_threshold_alone():
    infotext = rr.BaseInfotext.from_prompt("a cat", "").with_params(success_params())

    assert len(RE_PARAM.findall(infotext.split("\n")[-1])) >= 3


@pytest.mark.parametrize("seed", [0, 2 ** 32 - 1, -1])
@pytest.mark.parametrize("resolution", [512, 3840, 0])
def test_success_params_keep_values_as_given(seed, resolution):
    params = rr.seedvr2_generation_params("d", "v", resolution, seed)

    assert params["SeedVR2 Seed"] == seed and params["SeedVR2 Resolution"] == resolution


# --- plan_saving -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("is_txt2img, samples_save, disabled, requested",
                         list(itertools.product([True, False], repeat=4)))
def test_saving_rules_hold_for_every_combination(is_txt2img, samples_save, disabled, requested):
    plan = rr.plan_saving(is_txt2img, samples_save, disabled, requested)

    assert plan.save_results is (samples_save and not disabled)
    if plan.save_originals:
        assert requested and plan.save_results  # nothing at all is written with saving off
        assert not is_txt2img  # Forge saved the txt2img base images itself
    if disabled or not samples_save:
        assert plan.save_originals is False


@pytest.mark.parametrize("truthy, falsy", [(1, 0), ("yes", ""), ([1], []), (object(), None)])
def test_truthy_values_act_like_booleans(truthy, falsy):
    assert rr.plan_saving(False, truthy, falsy, truthy).save_originals
    assert not rr.plan_saving(False, truthy, truthy, truthy).save_originals
    assert not rr.plan_saving(True, truthy, falsy, truthy).save_originals


# --- merge_run_images ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad_index", [-1, -2, 10 ** 100, -(10 ** 100), 2 ** 63, "0", None, (0,), 0.5])
def test_indices_that_name_no_input_are_ignored(bad_index):
    merged = rr.merge_run_images(["a", "b"], {bad_index: "X"})

    assert merged == [rr.RunImage("a", False), rr.RunImage("b", False)]


def test_negative_index_does_not_replace_from_the_end():
    merged = rr.merge_run_images(["a", "b"], {-1: "X", 0: "A"})

    assert merged == [rr.RunImage("A", True), rr.RunImage("b", False)]


def test_large_batch_keeps_count_and_order():
    inputs = [f"in{index}" for index in range(5000)]
    upscaled = {index: f"out{index}" for index in range(0, 5000, 3)}

    merged = rr.merge_run_images(inputs, upscaled)

    assert len(merged) == 5000
    assert all(entry.is_upscaled == (index % 3 == 0) for index, entry in enumerate(merged))
    assert merged[4999] == rr.RunImage("in4999", False)
    assert merged[4998] == rr.RunImage("out4998", True)


def test_tuple_and_generator_inputs_are_merged():
    assert rr.merge_run_images(("a", "b"), {1: "B"}) == [rr.RunImage("a", False), rr.RunImage("B", True)]
    assert rr.merge_run_images((x for x in "ab"), {0: "A"}) == [rr.RunImage("A", True), rr.RunImage("b", False)]


def test_no_inputs_give_no_entries_even_with_upscaled_images():
    assert rr.merge_run_images([], {0: "X"}) == []


def test_the_same_input_object_twice_keeps_two_entries():
    image = object()

    merged = rr.merge_run_images([image, image], {1: "X"})

    assert merged == [rr.RunImage(image, False), rr.RunImage("X", True)]


def test_merge_does_not_mutate_its_arguments():
    inputs, upscaled = ["a", "b"], {0: "A"}

    rr.merge_run_images(inputs, upscaled)

    assert inputs == ["a", "b"] and upscaled == {0: "A"}

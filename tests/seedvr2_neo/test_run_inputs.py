"""Which images a run upscales, their pipeline mode, grouping and the seed — `seedvr2_neo/run_inputs.py`."""
import random

import pytest
from PIL import Image

from seedvr2_neo import run_inputs as ri


# --- select_generated_images ---

def test_grid_in_front_is_skipped():
    # Forge: batch of 4 with "return grid" → [grid, a, b, c, d], index_of_first_image = 1
    images = ["grid", "a", "b", "c", "d"]

    assert ri.select_generated_images(images, 1, 4) == ["a", "b", "c", "d"]


def test_single_image_without_grid_is_kept():
    assert ri.select_generated_images(["a"], 0, 1) == ["a"]


def test_images_after_the_batch_are_not_upscaled():
    assert ri.select_generated_images(["a", "b", "extra"], 0, 2) == ["a", "b"]


def test_fewer_images_than_seeds_gives_what_is_there():
    # a skipped or interrupted batch returns fewer images than all_seeds has entries
    assert ri.select_generated_images(["grid", "a"], 1, 4) == ["a"]


@pytest.mark.parametrize("first, count", [(None, 2), ("x", 2), (-3, 2), (True, 2)])
def test_unreadable_first_index_counts_as_zero(first, count):
    assert ri.select_generated_images(["a", "b"], first, count) == ["a", "b"]


@pytest.mark.parametrize("count", [None, "x", -1, 0])
def test_unreadable_count_gives_no_images(count):
    assert ri.select_generated_images(["a", "b"], 0, count) == []


def test_empty_result_gives_no_images():
    assert ri.select_generated_images([], 1, 4) == []


# --- select_distinct_images ---

def test_batch_tab_repeats_of_one_file_are_upscaled_once():
    image = Image.new("RGB", (4, 4))

    assert ri.select_distinct_images([image] * 3) == [image]


def test_equal_but_separate_images_are_all_kept_in_order():
    first, second = Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))

    assert ri.select_distinct_images([first, second, first]) == [first, second]


# --- to_pipeline_image ---

@pytest.mark.parametrize("mode", ["L", "1", "CMYK", "YCbCr", "P", "F"])
def test_modes_without_alpha_become_rgb(mode):
    assert ri.to_pipeline_image(Image.new(mode, (3, 2))).mode == "RGB"


@pytest.mark.parametrize("mode", ["LA", "PA", "RGBa", "La"])
def test_modes_with_alpha_become_rgba(mode):
    assert ri.to_pipeline_image(Image.new(mode, (3, 2))).mode == "RGBA"


def test_rgb_and_rgba_are_passed_unchanged():
    rgb, rgba = Image.new("RGB", (2, 2)), Image.new("RGBA", (2, 2))

    assert ri.to_pipeline_image(rgb) is rgb
    assert ri.to_pipeline_image(rgba) is rgba


def test_palette_with_transparency_keeps_alpha():
    image = Image.new("P", (2, 2))
    image.info["transparency"] = 0

    assert ri.to_pipeline_image(image).mode == "RGBA"


@pytest.mark.parametrize("mode", ["I;16", "I"])
def test_16_bit_grayscale_is_scaled_not_clipped(mode):
    image = Image.new(mode, (1, 1), 32768)  # mid-gray in 16 bits

    pixel = ri.to_pipeline_image(image).getpixel((0, 0))

    assert pixel[0] == pixel[1] == pixel[2]
    assert 120 <= pixel[0] <= 135


# --- group_by_shape ---

def test_same_size_images_form_one_group():
    images = [Image.new("RGB", (8, 8)) for _ in range(3)]

    assert ri.group_by_shape(images) == [[0, 1, 2]]


def test_other_size_or_mode_gets_its_own_group_in_first_seen_order():
    images = [Image.new("RGB", (8, 8)), Image.new("RGBA", (8, 8)), Image.new("RGB", (8, 8)),
              Image.new("RGB", (4, 8))]

    assert ri.group_by_shape(images) == [[0, 2], [1], [3]]


def test_no_images_no_groups():
    assert ri.group_by_shape([]) == []


# --- resolve_seed ---

def test_seed_typed_in_the_ui_wins():
    assert ri.resolve_seed(42, [7], 9) == 42


def test_random_ui_seed_follows_the_first_image():
    assert ri.resolve_seed(-1, [7, 8], 9) == 7


def test_without_image_seeds_the_run_seed_is_used():
    assert ri.resolve_seed(-1, None, 9) == 9


def test_all_random_picks_a_seed_in_the_int32_range():
    seed = ri.resolve_seed(-1, [-1], -1, rng=random.Random(1))

    assert 0 <= seed <= ri.MAX_RANDOM_SEED


def test_forge_seed_above_int32_is_accepted():
    # Forge picks its own seeds up to 2**32 - 1
    assert ri.resolve_seed(-1, [4_000_000_000], -1) == 4_000_000_000


@pytest.mark.parametrize("ui_seed", ["42", 42.0, " 42 "])
def test_api_seed_as_string_or_float_is_read(ui_seed):
    assert ri.resolve_seed(ui_seed, [], -1) == 42

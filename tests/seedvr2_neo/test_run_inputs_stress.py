"""Hostile images, indices and seeds for `seedvr2_neo/run_inputs.py`.

Images come from Forge (`processed.images`, `p.init_images`) and from API callers (any PNG/TIFF Pillow can open, so any
mode and any `info`); indices and seeds come from `Processed`, `p.all_seeds` and the UI / API `script_args`, so any type
can arrive. Rules checked here: the pipeline only ever gets RGB or RGBA, nothing raises on an odd value, a bad index
gives fewer images (never the grid), a bad seed means "not chosen".
"""
import io
import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest
from PIL import Image

from seedvr2_neo import run_inputs as ri

PIPELINE_MODES = {"RGB", "RGBA"}
ALL_MODES = ["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F", "LA", "PA", "RGBa", "La", "RGBX",
             "I;16", "I;16L", "I;16B", "I;16N"]
ALPHA_MODES = {"LA", "PA", "RGBa", "La", "RGBA"}


def png_roundtrip(image: Image.Image, **save_args) -> Image.Image:
    """The image as an API caller's PNG decodes in Forge: written and opened again, with the decoder's own `info`."""
    buffer = io.BytesIO()
    image.save(buffer, "PNG", **save_args)
    buffer.seek(0)
    reopened = Image.open(buffer)
    reopened.load()
    return reopened


# --- to_pipeline_image: every mode ---------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_pillow_mode_reaches_the_pipeline_as_rgb_or_rgba(mode):
    result = ri.to_pipeline_image(Image.new(mode, (3, 2)))

    assert result.mode in PIPELINE_MODES
    assert result.size == (3, 2)
    assert result.mode == ("RGBA" if mode in ALPHA_MODES else "RGB")


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_mode_with_transparency_info_becomes_rgba_or_stays_rgb(mode):
    # "CMYK with transparency info" and friends: Pillow must not refuse the RGBA conversion
    image = Image.new(mode, (3, 2))
    image.info["transparency"] = 0

    result = ri.to_pipeline_image(image)

    assert result.mode in PIPELINE_MODES


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("size", [(1, 1), (4096, 1), (1, 4096), (20_000, 2)])
def test_one_pixel_and_very_wide_images_keep_their_size(mode, size):
    result = ri.to_pipeline_image(Image.new(mode, size))

    assert result.size == size
    assert result.mode in PIPELINE_MODES


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_pipeline_modes_are_returned_as_the_same_object_even_with_odd_info(mode):
    image = Image.new(mode, (2, 2))
    image.info.update({"transparency": None, "dpi": "x", "": b"\0", "🧪": object(), "exif": b"garbage"})

    assert ri.to_pipeline_image(image) is image


def test_odd_info_keys_other_than_transparency_are_ignored():
    image = Image.new("L", (2, 2), 50)
    image.info.update({"dpi": (float("nan"), -1), "icc_profile": b"\xff" * 10, "gamma": "x", "Transparency": 0,
                       "": None, "parameters": "a\nb", "🧪": object()})

    result = ri.to_pipeline_image(image)

    assert result.mode == "RGB"  # "Transparency" with a capital T is not Pillow's key
    assert result.getpixel((0, 0)) == (50, 50, 50)


def test_input_image_is_not_modified():
    image = Image.new("P", (2, 2))
    image.info["transparency"] = 0
    before = (image.mode, image.size, dict(image.info), image.tobytes())

    ri.to_pipeline_image(image)

    assert (image.mode, image.size, dict(image.info), image.tobytes()) == before


# --- to_pipeline_image: pixel values -------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["I;16", "I;16L", "I;16B", "I;16N", "I"])
@pytest.mark.parametrize("value, low, high", [(65535, 255, 255), (32768, 120, 135), (257, 1, 1)])
def test_every_16_bit_layout_is_scaled_not_clipped(mode, value, low, high):
    pixel = ri.to_pipeline_image(Image.new(mode, (1, 1), value)).getpixel((0, 0))

    assert pixel[0] == pixel[1] == pixel[2]
    assert low <= pixel[0] <= high


def test_16_bit_png_from_a_file_is_scaled_not_clipped():
    image = png_roundtrip(Image.new("I;16", (2, 1), 32768))

    assert 120 <= ri.to_pipeline_image(image).getpixel((0, 0))[0] <= 135


@pytest.mark.xfail(strict=True, reason="known limitation: the tRNS key of a 16-bit grayscale PNG is dropped (the "
                                       "picture is scaled correctly but comes back opaque RGB); rare, see 07-RISKS")
def test_16_bit_png_with_transparency_is_scaled_and_keeps_alpha():
    source = Image.new("I;16", (2, 1), 32768)
    source.putpixel((1, 0), 1000)
    image = png_roundtrip(source, transparency=1000)
    assert image.mode == "I;16" and image.info.get("transparency") == 1000  # what Pillow gives Forge

    result = ri.to_pipeline_image(image)

    assert result.mode == "RGBA"
    assert 120 <= result.getpixel((0, 0))[0] <= 135
    assert result.getpixel((1, 0))[3] == 0


@pytest.mark.parametrize("value, expected", [(-1e30, 0), (-1.0, 0), (1e30, 255), (float("inf"), 255),
                                             (float("-inf"), 0), (128.0, 128)])
def test_float_images_with_extreme_values_are_clamped(value, expected):
    pixel = ri.to_pipeline_image(Image.new("F", (1, 1), value)).getpixel((0, 0))

    assert pixel == (expected,) * 3


def test_float_image_with_nan_does_not_raise():
    assert ri.to_pipeline_image(Image.new("F", (1, 1), float("nan"))).mode == "RGB"


@pytest.mark.parametrize("value, expected", [(-5, 0), (2 ** 31 - 1, 255)])
def test_32_bit_int_extremes_are_clamped(value, expected):
    assert ri.to_pipeline_image(Image.new("I", (1, 1), value)).getpixel((0, 0)) == (expected,) * 3


def test_palette_png_with_per_index_alpha_bytes_keeps_each_alpha():
    source = Image.new("P", (3, 1))
    source.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255])
    for x in range(3):
        source.putpixel((x, 0), x)
    image = png_roundtrip(source, transparency=b"\x00\x80\xff")

    result = ri.to_pipeline_image(image)

    assert result.mode == "RGBA"
    assert [result.getpixel((x, 0))[3] for x in range(3)] == [0, 128, 255]
    assert [result.getpixel((x, 0))[:3] for x in range(3)] == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def test_palette_with_transparency_bytes_set_in_memory_keeps_alpha():
    image = Image.new("P", (2, 1))
    image.putpalette([255, 0, 0, 0, 255, 0])
    image.putpixel((1, 0), 1)
    image.info["transparency"] = b"\x00\xff"

    result = ri.to_pipeline_image(image)

    assert (result.mode, result.getpixel((0, 0))[3], result.getpixel((1, 0))[3]) == ("RGBA", 0, 255)


def test_grayscale_png_with_transparent_value_becomes_rgba_with_that_pixel_clear():
    source = Image.new("L", (2, 1), 100)
    source.putpixel((1, 0), 7)

    result = ri.to_pipeline_image(png_roundtrip(source, transparency=7))

    assert result.getpixel((0, 0)) == (100, 100, 100, 255)
    assert result.getpixel((1, 0))[3] == 0


def test_rgb_png_with_transparent_colour_key_keeps_alpha():
    source = Image.new("RGB", (2, 1), (10, 20, 30))
    source.putpixel((1, 0), (1, 2, 3))
    image = png_roundtrip(source, transparency=(1, 2, 3))
    assert image.mode == "RGB" and image.info.get("transparency") == (1, 2, 3)

    result = ri.to_pipeline_image(image)

    assert result.mode == "RGBA"
    assert result.getpixel((1, 0))[3] == 0


@pytest.mark.parametrize("mode, fill", [("LA", (200, 128)), ("La", (100, 128)), ("RGBa", (100, 100, 100, 128))])
def test_alpha_modes_keep_a_half_transparent_pixel_half_transparent(mode, fill):
    # built with Image.new: Pillow's own RGBA -> La conversion drops alpha, which is not the case under test
    image = Image.new(mode, (1, 1), fill)

    alpha = ri.to_pipeline_image(image).getpixel((0, 0))[3]

    assert 126 <= alpha <= 130


def test_premultiplied_grayscale_is_unpremultiplied():
    # La (100, 128) is gray 200 at half alpha
    pixel = ri.to_pipeline_image(Image.new("La", (1, 1), (100, 128))).getpixel((0, 0))

    assert 197 <= pixel[0] <= 201 and pixel[0] == pixel[1] == pixel[2]


def test_palette_alpha_mode_keeps_its_alpha():
    image = Image.new("PA", (1, 1))
    image.putpalette([200, 200, 200])
    image.putpixel((0, 0), (0, 128))

    assert ri.to_pipeline_image(image).getpixel((0, 0)) == (200, 200, 200, 128)


def test_one_bit_image_is_black_and_white_rgb():
    image = Image.new("1", (2, 1), 0)
    image.putpixel((1, 0), 1)

    result = ri.to_pipeline_image(image)

    assert [result.getpixel((x, 0)) for x in range(2)] == [(0, 0, 0), (255, 255, 255)]


# --- group_by_shape --------------------------------------------------------------------------------------------------

def test_groups_cover_every_index_exactly_once_in_a_random_mix():
    rng = random.Random(7)
    images = [Image.new(rng.choice(["RGB", "RGBA"]), (rng.choice([1, 2, 3]), rng.choice([1, 2]))) for _ in range(300)]

    groups = ri.group_by_shape(images)

    assert sorted(index for group in groups for index in group) == list(range(300))
    for group in groups:
        assert group == sorted(group)
        assert len({(images[index].size, images[index].mode) for index in group}) == 1
    assert [group[0] for group in groups] == sorted(group[0] for group in groups)


def test_transposed_sizes_are_different_groups():
    images = [Image.new("RGB", (2, 3)), Image.new("RGB", (3, 2))]

    assert ri.group_by_shape(images) == [[0], [1]]


def test_the_same_object_repeated_is_grouped_by_index():
    image = Image.new("RGB", (2, 2))

    assert ri.group_by_shape([image, image, image]) == [[0, 1, 2]]


def test_tuple_and_generator_inputs_work_like_lists():
    images = [Image.new("RGB", (2, 2)), Image.new("RGBA", (2, 2))]

    assert ri.group_by_shape(tuple(images)) == [[0], [1]]
    assert ri.group_by_shape(image for image in images) == [[0], [1]]


# --- select_generated_images -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("first", [float("nan"), float("inf"), float("-inf"), "1.0", "one", b"x", [1], {}, object(),
                                   False, -(10 ** 100), -1.5])
def test_unreadable_or_negative_first_index_counts_as_zero(first):
    assert ri.select_generated_images(["a", "b"], first, 2) == ["a", "b"]


@pytest.mark.parametrize("first, expected", [(1.9, ["b", "c"]), ("1", ["b", "c"]), (" 2 ", ["c"]), (Decimal("1.5"), ["b", "c"]),
                                             (Fraction(5, 2), ["c"]), (10 ** 100, []), (2 ** 63, []), (3, [])])
def test_first_index_of_other_numeric_types_is_truncated(first, expected):
    assert ri.select_generated_images(["a", "b", "c"], first, 5) == expected


@pytest.mark.parametrize("count", [float("nan"), float("-inf"), True, False, "2.0", [], object(), -(10 ** 100)])
def test_unreadable_count_gives_nothing(count):
    assert ri.select_generated_images(["a", "b"], 0, count) == []


@pytest.mark.parametrize("count", [float("inf")])
def test_infinite_count_gives_nothing_rather_than_raising(count):
    assert ri.select_generated_images(["a", "b"], 0, count) == []


@pytest.mark.parametrize("count", [10 ** 100, 2 ** 63, 2 ** 64 + 1, 1e300])
def test_huge_count_gives_what_is_there(count):
    assert ri.select_generated_images(["grid", "a", "b"], 1, count) == ["a", "b"]


def test_grid_only_result_gives_nothing():
    assert ri.select_generated_images(["grid"], 1, 4) == []


def test_tuple_images_and_strings_give_a_list():
    assert ri.select_generated_images(("grid", "a"), 1, 1) == ["a"]


def test_result_is_a_new_list_not_a_view():
    images = ["a", "b"]

    selected = ri.select_generated_images(images, 0, 2)
    selected.append("x")

    assert images == ["a", "b"]


def test_selection_of_infotexts_uses_the_same_rule_as_images():
    # the script selects images and infotexts with the same index/count, so they stay paired
    images, infotexts = ["grid", "a", "b", "extra"], ["grid info", "a info", "b info", "extra info"]

    assert ri.select_generated_images(images, 1, 2) == ["a", "b"]
    assert ri.select_generated_images(infotexts, 1, 2) == ["a info", "b info"]


# --- select_distinct_images ------------------------------------------------------------------------------------------

def test_many_repeats_of_a_few_objects_keep_first_seen_order():
    a, b, c = (Image.new("RGB", (1, 1)) for _ in range(3))

    assert ri.select_distinct_images([b, a, b, c, a] * 1000) == [b, a, c]


def test_distinct_selection_accepts_tuples_and_generators():
    a, b = Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))

    assert ri.select_distinct_images((a, b, a)) == [a, b]
    assert ri.select_distinct_images(image for image in (a, a, b)) == [a, b]


def test_unhashable_items_are_still_deduplicated_by_identity():
    first, second = [], []

    assert ri.select_distinct_images([first, first, second]) == [first, second]


def test_empty_input_gives_empty_list():
    assert ri.select_distinct_images([]) == []
    assert ri.select_distinct_images(()) == []


# --- resolve_seed ----------------------------------------------------------------------------------------------------

NOT_CHOSEN = [None, -1, "-1", -1.0, float("nan"), float("inf"), float("-inf"), "nan", "inf", "", " ", "abc", "0x10",
              "🌱", True, False, [], [42], {}, {"seed": 42}, object(), b"42x", 10 ** 400, -(10 ** 400), "1e400",
              ri.MAX_SEED + 1, -2, -1e30, 1j]


@pytest.mark.parametrize("ui_seed", NOT_CHOSEN)
def test_unusable_ui_seed_falls_back_to_the_image_seed(ui_seed):
    assert ri.resolve_seed(ui_seed, [7], 9) == 7


@pytest.mark.parametrize("bad", NOT_CHOSEN)
def test_unusable_everywhere_gives_a_random_seed_in_range(bad):
    seed = ri.resolve_seed(bad, [bad], bad, rng=random.Random(3))

    assert isinstance(seed, int) and not isinstance(seed, bool)
    assert 0 <= seed <= ri.MAX_RANDOM_SEED


@pytest.mark.parametrize("ui_seed, expected", [
    (0, 0), ("0", 0), (ri.MAX_SEED, ri.MAX_SEED), (float(ri.MAX_SEED), ri.MAX_SEED), ("4294967295", ri.MAX_SEED),
    ("1e3", 1000), (" 42\n", 42), ("+42", 42), (42.9, 42), (Decimal("42.7"), 42), (Fraction(85, 2), 42),
    (b"42", 42), (-0.5, 0),
])
def test_ui_seed_of_other_numeric_forms_is_read(ui_seed, expected):
    assert ri.resolve_seed(ui_seed, [7], 9) == expected


def test_seed_is_always_a_plain_int():
    for ui_seed in (42.0, "42", Decimal(42), b"42"):
        seed = ri.resolve_seed(ui_seed, None, None)

        assert type(seed) is int


@pytest.mark.parametrize("image_seeds", [None, [], (), "", {}, iter([5]), (seed for seed in [5]), 5, object()])
def test_image_seeds_that_are_not_a_usable_sequence_fall_back_to_the_run_seed(image_seeds):
    assert ri.resolve_seed(-1, image_seeds, 9) == 9


def test_image_seeds_as_a_tuple_use_its_first_entry():
    assert ri.resolve_seed(-1, (11, 12), 9) == 11


def test_only_the_first_image_seed_is_considered():
    # a bad first entry falls through to the run seed, not to the second image seed
    assert ri.resolve_seed(-1, [-1, 12], 9) == 9


def test_random_seed_comes_from_the_given_rng():
    first = ri.resolve_seed(-1, None, -1, rng=random.Random(5))
    second = ri.resolve_seed(-1, None, -1, rng=random.Random(5))

    assert first == second


def test_random_seed_does_not_touch_the_global_random_state_when_an_rng_is_given():
    random.seed(123)
    expected_next = random.random()
    random.seed(123)

    ri.resolve_seed(-1, None, -1, rng=random.Random(1))

    assert random.random() == expected_next


def test_nan_seed_is_not_mistaken_for_zero():
    assert not math.isnan(ri.resolve_seed(float("nan"), None, 9))
    assert ri.resolve_seed(float("nan"), None, 9) == 9

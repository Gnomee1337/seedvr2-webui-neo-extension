"""The output size shown before a run — `seedvr2_neo/size_preview.py`."""
import pytest

from seedvr2_neo import size_preview as sp


# --- describe_output_size ---

def test_scale_factor_names_the_final_size_and_the_source():
    assert sp.describe_output_size((512, 384), 1080, "Scale factor", 2, 2048) == \
        "Output size: **1024×768** from 512×384"


def test_shortest_edge_and_long_edge_modes_are_described_too():
    assert sp.describe_output_size((1920, 1080), 1080, "Shortest edge", 2, 2048) == \
        "Output size: **1920×1080** from 1920×1080"
    assert sp.describe_output_size((1920, 1080), 1080, "Long edge", 2, 2048) == \
        "Output size: **2048×1152** from 1920×1080"


def test_the_preview_matches_the_run_cap_at_3840():
    assert "**2560×3840**" in sp.describe_output_size((1024, 1536), 1080, "Scale factor", 4, 2048)


def test_a_raise_to_512_is_named():
    assert sp.describe_output_size((384, 384), 1080, "Scale factor", 1, 2048) == \
        "Output size: **512×512** from 384×384 (shortest edge raised to 512 px)"


@pytest.mark.parametrize("size", [None, (0, 0), (512, 0), (-1, 512), ("a", 512), (512,), "512x384", (float("nan"), 1),
                                  (float("inf"), 1)])
def test_unknown_or_broken_source_sizes_give_the_generic_note(size):
    assert sp.describe_output_size(size, 1080, "Scale factor", 2, 2048) == sp.UNKNOWN_SOURCE_NOTE


@pytest.mark.parametrize("mode, factor, edge", [("8K", "x", None), (None, float("nan"), "abc"), (3, -5, 10 ** 9)])
def test_hostile_control_values_are_read_like_the_run_reads_them(mode, factor, edge):
    assert sp.describe_output_size((512, 384), 1080, mode, factor, edge).startswith("Output size: **")


# --- txt2img_final_size ---

def test_without_hires_fix_the_size_is_width_and_height():
    assert sp.txt2img_final_size(832, 1216, False, 2.0, 0, 0) == (832, 1216)


@pytest.mark.parametrize("width, height, scale, resize_x, resize_y, step, expected", [
    (512, 512, 2.0, 0, 0, 64, (1024, 1024)),
    (832, 1216, 1.5, 0, 0, 64, (1280, 1856)),  # 1248 and 1824 are halves of 64: rounded up
    (832, 1216, 1.5, 0, 0, 8, (1248, 1824)),
    (512, 768, 2.0, 1024, 0, 64, (1024, 1536)),  # one side set keeps the aspect ratio; the scale is ignored
    (512, 768, 2.0, 0, 1024, 64, (704, 1024)),  # 682.7 rounded to 64
    (512, 768, 2.0, 1000, 1000, 64, (1024, 1024)),  # both set are taken as they are, rounded
])
def test_hires_fix_size_follows_forge(width, height, scale, resize_x, resize_y, step, expected):
    assert sp.txt2img_final_size(width, height, True, scale, resize_x, resize_y, step) == expected


@pytest.mark.parametrize("args", [(0, 512, False, 2, 0, 0), (512, None, False, 2, 0, 0), ("a", 512, True, 2, 0, 0),
                                  (512, 512, True, None, 0, 0), (512, 512, True, float("nan"), 0, 0),
                                  (512, 512, True, float("inf"), 0, 0), (512, 512, True, 0, 0, 0)])
def test_unreadable_values_give_no_size(args):
    assert sp.txt2img_final_size(*args) is None


@pytest.mark.parametrize("step", [0, None, -8, "abc", float("nan")])
def test_a_broken_res_step_falls_back_to_64(step):
    assert sp.txt2img_final_size(832, 1216, True, 1.5, 0, 0, step) == (1280, 1856)

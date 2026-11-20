"""Hostile values for the output size modes through the Forge script `scripts/seedvr2_script.py`, against the stubs of
`forge_stubs.py`.

API callers send `script_args` positionally: older callers send 13 or 16 values, newer ones any value in the three size
slots. The preview handlers get whatever Gradio or the browser sends. Checked here: no value crashes a run or turns into
an error infotext, the pipeline never gets a size outside the caps, old callers keep the shortest edge, the preview
names the size the run uses, and nothing of one run or one tab leaks into another.
"""
import sys

import pytest
from PIL import Image

from forge_stubs import make_img2img, run_args

GARBAGE = [None, "", "   ", "8K", "long_edge", "Long edge, Steps: 999", "Scale factor\nSteps: 1", "💥" * 2000,
           float("nan"), float("inf"), float("-inf"), -1, 0, 10 ** 400, 1e308, True, False, [], {}, b"Long edge",
           object(), complex(1, 2)]
MODES = ["Scale factor", "Shortest edge", "Long edge"]


def short_id(value):
    return repr(value)[:30]


def mixed_inputs():
    return [Image.new("RGB", (64, 48)), Image.new("RGB", (48, 64)), Image.new("RGB", (64, 48)),
            Image.new("RGB", (400, 100)), Image.new("RGB", (1, 1))]


def assert_clean_run(processed, pipeline, image_count, capped=True):
    """No error, one size mode per infotext, and the pipeline got sizes inside the caps. In the capped modes a very wide
    panorama may keep a shortest edge under 512: raising it further would pass the long edge cap."""
    assert len(processed.images) == len(processed.infotexts) == image_count
    assert not any("SeedVR2 Error" in infotext for infotext in processed.infotexts)
    assert all(infotext.count("SeedVR2 Size Mode: ") == 1 for infotext in processed.infotexts)
    for kwargs in pipeline.generation_kwargs:
        assert (1 if capped else 512) <= kwargs["resolution"] <= 3840
        if capped:
            assert kwargs["max_resolution"] <= 3840


# --- garbage in the new arguments ---------------------------------------------------------------------------------

@pytest.mark.parametrize("argument", ["size_mode", "scale_factor", "long_edge"])
@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_in_one_size_argument_never_fails_the_run(script, pipeline, argument, value):
    base = {"size_mode": "Scale factor", "scale_factor": 2.0, "long_edge": 2048}
    # a fixed seed: a random one could itself contain "999"
    processed = script.run(make_img2img(mixed_inputs()), **run_args(**{**base, argument: value, "seed": 42}))
    assert_clean_run(processed, pipeline, 5, capped=argument != "size_mode")
    assert "Steps: 999" not in "".join(processed.infotexts) and "Steps: 1" not in "".join(processed.infotexts)


@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_garbage_in_every_size_argument_at_once_falls_back_to_the_shortest_edge(script, pipeline, value):
    processed = script.run(make_img2img(mixed_inputs()),
                           **run_args(size_mode=value, scale_factor=value, long_edge=value, resolution=value))
    assert_clean_run(processed, pipeline, 5, capped=False)
    assert all("SeedVR2 Size Mode: Shortest edge" in infotext for infotext in processed.infotexts)
    assert len({kwargs["resolution"] for kwargs in pipeline.generation_kwargs}) == 1  # one shortest edge for all


@pytest.mark.parametrize("mode", MODES)
def test_extreme_aspect_ratios_from_the_api_reach_the_pipeline_inside_the_caps(script, pipeline, mode):
    inputs = [Image.new("RGB", (4000, 250)), Image.new("RGB", (250, 4000)), Image.new("RGB", (3, 2000)),
              Image.new("RGB", (900, 900))]
    processed = script.run(make_img2img(inputs), **run_args(size_mode=mode, scale_factor=4.0, long_edge=3840))
    assert_clean_run(processed, pipeline, 4, capped=mode != "Shortest edge")


# --- older API callers -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("count", [13, 16])
def test_old_positional_calls_keep_the_shortest_edge(script, pipeline, count):
    args = list(run_args(resolution=640).values())[:count]
    processed = script.run(make_img2img([Image.new("RGB", (64, 48)), Image.new("RGB", (300, 200))]), *args)
    assert [kwargs["resolution"] for kwargs in pipeline.generation_kwargs] == [640, 640]
    assert all("SeedVR2 Size Mode: Shortest edge" in infotext and "SeedVR2 Scale" not in infotext
               for infotext in processed.infotexts)


def test_a_call_with_only_the_mode_uses_the_default_factor(script, pipeline):
    args = list(run_args(size_mode="Scale factor").values())[:17]
    processed = script.run(make_img2img([Image.new("RGB", (600, 400))]), *args)
    assert pipeline.generation_kwargs[0]["resolution"] == 800  # x2, the slider default
    assert "SeedVR2 Scale: 2.0" in processed.infotexts[0]


def test_a_call_with_the_mode_and_factor_but_no_long_edge_works(script, pipeline):
    args = list(run_args(size_mode="Long edge", scale_factor=3.0).values())[:18]
    processed = script.run(make_img2img([Image.new("RGB", (1000, 500))]), *args)
    assert pipeline.generation_kwargs[0]["resolution"] == 1024  # long edge 2048, the slider default
    assert "SeedVR2 Long Edge: 2048" in processed.infotexts[0]


def test_string_values_as_a_json_client_may_send_them_are_read(script, pipeline):
    processed = script.run(make_img2img([Image.new("RGB", (600, 400))]),
                           **run_args(size_mode=" long EDGE ", scale_factor="3", long_edge="1536"))
    assert pipeline.generation_kwargs[0]["resolution"] == 1024
    assert "SeedVR2 Size Mode: Long edge, SeedVR2 Long Edge: 1536" in processed.infotexts[0]


# --- one run after another ---------------------------------------------------------------------------------------

def test_runs_with_different_modes_do_not_leak_into_each_other(script, pipeline):
    inputs = [Image.new("RGB", (600, 400))]
    expected = []
    for mode, factor, edge, resolution in [("Scale factor", 4.0, 2048, 1080), ("Long edge", 2.0, 1024, 1080),
                                           ("Shortest edge", 3.0, 3840, 640), ("Scale factor", 1.5, 3840, 3840),
                                           ("garbage", 4.0, 3840, 720), ("Scale factor", 4.0, 2048, 1080)]:
        before = len(pipeline.generation_kwargs)
        processed = script.run(make_img2img(inputs), **run_args(size_mode=mode, scale_factor=factor, long_edge=edge,
                                                                resolution=resolution))
        expected.append(pipeline.generation_kwargs[before]["resolution"])
        assert len(pipeline.generation_kwargs) == before + 1
        assert f"SeedVR2 Resolution: {expected[-1]}" in processed.infotexts[0]
    assert expected == [1600, 683, 640, 600, 720, 1600]


def test_the_txt2img_tab_and_the_img2img_tab_scripts_keep_their_own_size_sources(script_module):
    txt2img, img2img = script_module.Script(), script_module.Script()
    gr = sys.modules["gradio"]
    for elem_id in ("txt2img_width", "txt2img_height", "txt2img_hr-checkbox", "txt2img_hr_scale",
                    "txt2img_hr_resize_x", "txt2img_hr_resize_y"):
        txt2img.after_component(gr.Slider(value=512, elem_id=elem_id), elem_id=elem_id)
    for elem_id in ("img2img_width", "img2img_height", "img2img_hr_scale"):
        img2img.after_component(gr.Slider(value=512, elem_id=elem_id), elem_id=elem_id)
    assert txt2img._txt2img_preview_inputs([]) is not None
    assert img2img._txt2img_preview_inputs([]) is None
    assert script_module.Script()._txt2img_size_sources == {}


@pytest.mark.parametrize("kwargs", [{}, {"elem_id": None}, {"elem_id": ""}, {"elem_id": 5}, {"elem_id": "txt2img"},
                                    {"elem_id": "TXT2IMG_WIDTH"}, {"elem_id": "txt2img_width "},
                                    {"elem_id": ["txt2img_width"]}, {"elem_id": "txt2img_hr"}])
def test_components_that_are_not_forge_s_size_controls_are_ignored(script, kwargs):
    gr = sys.modules["gradio"]
    script.after_component(gr.Slider(value=1, **kwargs), **kwargs)
    assert script._txt2img_size_sources == {}


def test_a_rebuilt_ui_uses_the_newest_forge_controls(script):
    gr = sys.modules["gradio"]
    for value in (512, 768):
        for elem_id in ("txt2img_width", "txt2img_height", "txt2img_hr-checkbox", "txt2img_hr_scale",
                        "txt2img_hr_resize_x", "txt2img_hr_resize_y"):
            script.after_component(gr.Slider(value=value, elem_id=elem_id), elem_id=elem_id)
    inputs = script._txt2img_preview_inputs([])
    assert [component.value for component in inputs[:2]] == [768, 768]


# --- preview handlers ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_the_txt2img_preview_handler_never_raises(script_module, forge, value):
    for position in range(10):
        args = [512, 384, True, 2.0, 0, 0, 1080, "Scale factor", 2.0, 2048]
        args[position] = value
        assert script_module.preview_txt2img_output_size(*args).startswith("Output size")


@pytest.mark.parametrize("res_step", [None, "abc", 0, -8, float("nan"), 10 ** 30])
def test_the_txt2img_preview_survives_an_odd_res_step_option(script_module, forge, res_step):
    forge.opts.res_step = res_step
    assert script_module.preview_txt2img_output_size(512, 384, True, 2.0, 0, 0, 1080, "Scale factor", 2.0,
                                                     2048).startswith("Output size")


@pytest.mark.parametrize("value", GARBAGE, ids=short_id)
def test_the_img2img_preview_handler_never_raises(script_module, value):
    for position in range(6):
        if position < 2 and isinstance(value, int) and value > 10 ** 300:
            continue  # the overflow of a huge source size is the strict xfail in test_size_preview_stress.py
        args = [512, 384, 1080, "Long edge", 2.0, 2048]
        args[position] = value
        assert script_module.preview_img2img_output_size(*args).startswith("Output size")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [(600, 400), (64, 48), (400, 100), (1, 1), (1500, 1000)])
def test_the_img2img_preview_names_the_size_the_run_plans(script, script_module, pipeline, capsys, mode, size):
    line = script_module.preview_img2img_output_size(*size, 1080, mode, 1.7, 1600)
    script.run(make_img2img([Image.new("RGB", size)]),
               **run_args(size_mode=mode, scale_factor=1.7, long_edge=1600, resolution=1080))
    planned = capsys.readouterr().out.split(f"{size[0]}x{size[1]} -> ")[1].split()[0]
    assert f"**{planned.replace('x', '×')}**" in line


@pytest.mark.parametrize("mode", MODES)
def test_the_txt2img_preview_names_the_size_the_run_plans(script, script_module, make_txt2img, pipeline, capsys, mode):
    line = script_module.preview_txt2img_output_size(320, 192, False, 2.0, 0, 0, 1080, mode, 3.3, 1344)
    script.run(make_txt2img(width=320, height=192), **run_args(size_mode=mode, scale_factor=3.3, long_edge=1344))
    planned = capsys.readouterr().out.split("320x192 -> ")[1].split()[0]
    assert f"**{planned.replace('x', '×')}**" in line


@pytest.mark.parametrize("value", GARBAGE + MODES, ids=short_id)
def test_the_mode_switch_always_shows_exactly_one_control(script, value):
    updates = script._show_size_controls(value)
    assert len(updates) == 3 and sum(bool(update["visible"]) for update in updates) == 1

"""Raw UI / API values → upscale settings inside the UI's ranges — `seedvr2_neo/upscale_settings.py`."""
import pytest

from seedvr2_neo import upscale_settings as us


def parse(**overrides):
    values = dict(resolution=1080, input_noise=0.0, latent_noise=0.0, use_tile_vae=True, tile_size=512,
                  tile_overlap=128)
    values.update(overrides)
    return us.parse_upscale_settings(**values)


def test_ui_values_pass_unchanged():
    assert parse() == us.UpscaleSettings(1080, 0.0, 0.0, True, 512, 128)


@pytest.mark.parametrize("raw, expected", [(100000, 3840), (100, 512), ("2048", 2048), (1080.7, 1080)])
def test_resolution_is_kept_in_the_slider_range(raw, expected):
    assert parse(resolution=raw).resolution == expected


@pytest.mark.parametrize("raw", ["abc", None, float("nan"), float("inf"), True, [1080]])
def test_unreadable_resolution_takes_the_default(raw):
    assert parse(resolution=raw).resolution == us.RESOLUTION.default


@pytest.mark.parametrize("raw, expected", [(-1, 0.0), (5, 1.0), ("0.5", 0.5)])
def test_noise_is_kept_between_0_and_1(raw, expected):
    assert parse(input_noise=raw).input_noise == expected
    assert parse(latent_noise=raw).latent_noise == expected


@pytest.mark.parametrize("raw, expected", [(0, 512), (99999, 2048)])
def test_tile_size_is_clamped(raw, expected):
    assert parse(tile_size=raw).tile_size == expected


@pytest.mark.parametrize("raw", ["abc", None, float("nan")])
def test_unreadable_tile_size_takes_the_512_default(raw):
    """512 is the size with which the VAE phases finish on 8 GB of VRAM."""
    assert parse(tile_size=raw).tile_size == 512


def test_tile_overlap_is_clamped():
    assert parse(tile_overlap=0).tile_overlap == 64
    assert parse(tile_size=2048, tile_overlap=10000).tile_overlap == 512


@pytest.mark.parametrize("tile, overlap, expected", [(512, 512, 256), (512, 384, 256), (512, 256, 256), (1024, 512, 512),
                                                     ("abc", 512, 256)])
def test_tile_overlap_is_capped_at_half_the_tile(tile, overlap, expected):
    """An overlap saved next to the old 1024 tile must not pair with 512 into a stride of a few pixels."""
    assert parse(tile_size=tile, tile_overlap=overlap).tile_overlap == expected


def test_longest_edge_is_capped_at_four_times_the_resolution():
    # a 1x4096 API input must not ask for a gigapixel output
    assert parse(resolution=1080).target_for((1, 4096)).max_resolution == 4320
    assert parse(resolution=100000).target_for((1, 4096)).max_resolution == 4 * 3840


@pytest.mark.parametrize("raw, expected", [("False", False), ("true", True), (0, False), (None, True)])
def test_tiling_flag_reads_api_strings(raw, expected):
    assert parse(use_tile_vae=raw).use_tile_vae is expected


# --- size modes ---

def test_old_callers_get_the_shortest_edge_mode():
    settings = parse()

    assert settings.size_mode is us.SizeMode.SHORTEST_EDGE
    assert settings.target_for((512, 384)) == us.OutputTarget(1080, 4320, False)
    assert settings.target_for((4000, 3000)).resolution == 1080  # the same for every input size, as before


@pytest.mark.parametrize("raw, expected", [
    ("Scale factor", us.SizeMode.SCALE_FACTOR), ("  long EDGE ", us.SizeMode.LONG_EDGE),
    (us.SizeMode.LONG_EDGE, us.SizeMode.LONG_EDGE), ("Shortest edge", us.SizeMode.SHORTEST_EDGE),
    ("8K", us.SizeMode.SHORTEST_EDGE), ("", us.SizeMode.SHORTEST_EDGE), (None, us.SizeMode.SHORTEST_EDGE),
    (1, us.SizeMode.SHORTEST_EDGE), (["Scale factor"], us.SizeMode.SHORTEST_EDGE), (True, us.SizeMode.SHORTEST_EDGE),
])
def test_size_mode_is_read_leniently_and_falls_back_to_the_default(raw, expected):
    assert parse(size_mode=raw).size_mode is expected


@pytest.mark.parametrize("raw, expected", [(2, 2.0), ("1.5", 1.5), (0.5, 1.0), (-3, 1.0), (10, 4.0), (2.345, 2.35),
                                           ("abc", 2.0), (None, 2.0), (float("nan"), 2.0), (float("inf"), 2.0),
                                           (True, 2.0)])
def test_scale_factor_is_kept_between_1_and_4(raw, expected):
    assert parse(scale_factor=raw).scale_factor == expected


@pytest.mark.parametrize("raw, expected", [(2048, 2048), ("3000", 3000), (100, 1024), (99999, 3840), (-1, 1024),
                                           ("abc", 2048), (None, 2048), (float("nan"), 2048)])
def test_long_edge_is_kept_in_its_slider_range(raw, expected):
    assert parse(long_edge=raw).long_edge == expected


def factor(value):
    return parse(size_mode="Scale factor", scale_factor=value)


def long_edge(value):
    return parse(size_mode="Long edge", long_edge=value)


@pytest.mark.parametrize("settings, size, expected", [
    (factor(2), (512, 384), (1024, 768)),
    (factor(1.5), (1024, 1024), (1536, 1536)),
    (factor(4), (768, 512), (3072, 2048)),
    (factor(1), (1920, 1080), (1920, 1080)),
    (factor(1.5), (1023, 767), (1532, 1150)),  # torchvision truncates the long edge, then both go down to even
    (long_edge(2048), (1920, 1080), (2048, 1152)),
    (long_edge(2048), (512, 384), (2048, 1536)),
    (long_edge(2048), (384, 512), (1536, 2048)),  # portrait
    (long_edge(3840), (1024, 1024), (3840, 3840)),
])
def test_output_size_in_the_scale_factor_and_long_edge_modes(settings, size, expected):
    assert settings.target_for(size).output_size(size) == expected


@pytest.mark.parametrize("settings", [factor(4), long_edge(3840), long_edge(99999)])
@pytest.mark.parametrize("size", [(1024, 1536), (2048, 2048), (3840, 2160), (8000, 2000), (16000, 1000), (1, 4096)])
def test_no_new_mode_gives_a_long_edge_over_3840(settings, size):
    """6K / 8K need DiT tiling; on an 8 GB GPU they would run for a very long time or run out of memory."""
    assert max(settings.target_for(size).output_size(size)) <= us.MAX_OUTPUT_LONG_EDGE


def test_factor_4_on_a_tall_image_is_capped_at_3840_on_the_long_edge():
    assert factor(4).target_for((1024, 1536)).output_size((1024, 1536)) == (2560, 3840)


@pytest.mark.parametrize("settings, size", [(factor(1), (512, 384)), (factor(1.2), (384, 384)),
                                            (factor(4), (100, 100))])
def test_outputs_under_512_are_raised_to_512_and_flagged(settings, size):
    target = settings.target_for(size)

    assert target.resolution == 512 and target.raised_to_minimum
    assert min(target.output_size(size)) == 512


@pytest.mark.parametrize("settings, size, expected", [
    (long_edge(1024), (1024, 256), (1024, 256)),  # 512 would make the long edge 2048, past the chosen 1024
    (factor(2), (4096, 512), (3840, 480)),  # the 3840 cap wins over the factor and over the 512 minimum
    (factor(2), (2048, 448), (3840, 840)),  # wider than 4:1 keeps the full long edge
    (factor(1), (600, 3600), (600, 3600)),  # x1 of a tall image is not made smaller
    (long_edge(3840), (768, 3840), (768, 3840)),
])
def test_wide_panoramas_get_the_long_edge_cap_not_a_smaller_image(settings, size, expected):
    target = settings.target_for(size)

    assert target.output_size(size) == expected
    assert target.raise_note() == ""


def test_the_long_edge_mode_never_raises_since_its_cap_is_the_chosen_long_edge():
    assert not long_edge(1024).target_for((1024, 256)).raised_to_minimum


def test_a_raise_stopped_by_the_cap_names_the_edge_it_reached():
    target = factor(1).target_for((100, 1000))

    assert target.output_size((100, 1000)) == (384, 3840)
    assert target.raise_note() == " (shortest edge raised to 384 px)"


def test_an_output_of_512_or_more_is_not_flagged():
    assert not factor(1).target_for((512, 512)).raised_to_minimum


def test_shortest_edge_mode_ignores_the_factor_and_long_edge():
    settings = parse(resolution=1536, scale_factor=4, long_edge=3840)

    assert settings.target_for((512, 384)).output_size((512, 384)) == (2048, 1536)


@pytest.mark.parametrize("size", [(0, 0), (-5, 100), (1, 1)])
def test_degenerate_sizes_do_not_crash(size):
    for settings in (parse(), factor(4), long_edge(3840)):
        width, height = settings.target_for(size).output_size(size)
        assert width >= 0 and height >= 0

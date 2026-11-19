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
    assert parse(resolution=1080).max_resolution == 4320
    assert parse(resolution=100000).max_resolution == 4 * 3840


@pytest.mark.parametrize("raw, expected", [("False", False), ("true", True), (0, False), (None, True)])
def test_tiling_flag_reads_api_strings(raw, expected):
    assert parse(use_tile_vae=raw).use_tile_vae is expected

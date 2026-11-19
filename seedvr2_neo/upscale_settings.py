"""The numeric upscale settings of one run, read from raw UI / API values and kept inside the UI's ranges.

The script (`scripts/seedvr2_script.py`) builds its sliders from the same bounds, so an API caller cannot ask for more
than the UI allows (an unbounded resolution means unbounded memory). Values are parsed before any generation, so a
bad API value costs nothing. No Forge imports. Memory settings (After run, BlockSwap): `seedvr2_neo/memory_plan.py`.
"""

import math
from dataclasses import dataclass
from typing import Any

from seedvr2_neo.memory_plan import parse_flag


@dataclass(frozen=True)
class Bounds:
    """A slider's range and default; `step` is the UI's step, not enforced on API values."""

    minimum: float
    maximum: float
    default: float
    step: float


RESOLUTION = Bounds(512, 3840, 1080, 64)
NOISE = Bounds(0.0, 1.0, 0.0, 0.05)
# 512 keeps VAE encoding and decoding inside 8 GB of VRAM; with 1024 and `--cuda-malloc` a phase could run for many
# minutes at full VRAM.
TILE_SIZE = Bounds(512, 2048, 512, 128)
TILE_OVERLAP = Bounds(64, 512, 128, 32)
USE_TILE_VAE_DEFAULT = True

# The longest output edge is capped at this many times the shortest one: a very thin input (1x4096 through the API)
# would otherwise ask for a gigapixel output. Up to 4:1 panoramas keep the requested resolution.
MAX_LONG_EDGE_FACTOR = 4


@dataclass(frozen=True)
class UpscaleSettings:
    """What the pipeline gets besides the models and the seed."""

    resolution: int
    input_noise: float
    latent_noise: float
    use_tile_vae: bool
    tile_size: int
    tile_overlap: int

    @property
    def max_resolution(self) -> int:
        """The pipeline's `max_resolution` (longest edge); it scales the whole image down when exceeded."""
        return self.resolution * MAX_LONG_EDGE_FACTOR


def parse_upscale_settings(resolution: Any, input_noise: Any, latent_noise: Any, use_tile_vae: Any, tile_size: Any,
                           tile_overlap: Any) -> UpscaleSettings:
    """Raw values → settings. Out-of-range values are clamped to the slider's range; unreadable ones (text, NaN,
    None) take the slider's default. The tile overlap is also capped at half the tile size."""
    tile = int(clamp_to_bounds(tile_size, TILE_SIZE))
    return UpscaleSettings(
        resolution=int(clamp_to_bounds(resolution, RESOLUTION)),
        input_noise=clamp_to_bounds(input_noise, NOISE),
        latent_noise=clamp_to_bounds(latent_noise, NOISE),
        use_tile_vae=USE_TILE_VAE_DEFAULT if use_tile_vae is None else parse_flag(use_tile_vae),
        tile_size=tile,
        # The VAE steps by tile minus overlap: an overlap near the tile size (512 with 512, both slider maximums allow
        # it) means thousands of tiles and a phase that looks frozen. Half the tile keeps it to a few times the tiles.
        tile_overlap=min(int(clamp_to_bounds(tile_overlap, TILE_OVERLAP)), tile // 2),
    )


def clamp_to_bounds(value: Any, bounds: Bounds) -> float:
    """`value` as a number inside `bounds`, or the default when it is not a finite number."""
    if isinstance(value, bool):
        return bounds.default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return bounds.default
    if not math.isfinite(number):
        return bounds.default
    return min(max(number, bounds.minimum), bounds.maximum)

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
TILE_SIZE = Bounds(512, 2048, 1024, 128)
TILE_OVERLAP = Bounds(64, 512, 128, 32)
USE_TILE_VAE_DEFAULT = True


@dataclass(frozen=True)
class UpscaleSettings:
    """What the pipeline gets besides the models and the seed."""

    resolution: int
    input_noise: float
    latent_noise: float
    use_tile_vae: bool
    tile_size: int
    tile_overlap: int


def parse_upscale_settings(resolution: Any, input_noise: Any, latent_noise: Any, use_tile_vae: Any, tile_size: Any,
                           tile_overlap: Any) -> UpscaleSettings:
    """Raw values → settings. Out-of-range values are clamped to the slider's range; unreadable ones (text, NaN,
    None) take the slider's default."""
    return UpscaleSettings(
        resolution=int(clamp_to_bounds(resolution, RESOLUTION)),
        input_noise=clamp_to_bounds(input_noise, NOISE),
        latent_noise=clamp_to_bounds(latent_noise, NOISE),
        use_tile_vae=USE_TILE_VAE_DEFAULT if use_tile_vae is None else parse_flag(use_tile_vae),
        tile_size=int(clamp_to_bounds(tile_size, TILE_SIZE)),
        tile_overlap=int(clamp_to_bounds(tile_overlap, TILE_OVERLAP)),
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

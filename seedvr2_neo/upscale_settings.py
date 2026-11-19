"""The numeric upscale settings of one run, read from raw UI / API values and kept inside the UI's ranges.

The script (`scripts/seedvr2_script.py`) builds its sliders from the same bounds, so an API caller cannot ask for more
than the UI allows (an unbounded resolution means unbounded memory). Values are parsed before any generation, so a
bad API value costs nothing. No Forge imports. Memory settings (After run, BlockSwap): `seedvr2_neo/memory_plan.py`.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

from seedvr2_neo.memory_plan import parse_flag


@dataclass(frozen=True)
class Bounds:
    """A slider's range and default; `step` is the UI's step, not enforced on API values."""

    minimum: float
    maximum: float
    default: float
    step: float


RESOLUTION = Bounds(512, 3840, 1080, 64)
SCALE_FACTOR = Bounds(1.0, 4.0, 2.0, 0.1)
LONG_EDGE = Bounds(1024, 3840, 2048, 64)
NOISE = Bounds(0.0, 1.0, 0.0, 0.05)
# 512 lets the VAE phases finish on 8 GB of VRAM (without `--cuda-malloc` with no spill into shared memory); with
# 1024 and `--cuda-malloc` a phase could run for many minutes at full VRAM.
TILE_SIZE = Bounds(512, 2048, 512, 128)
TILE_OVERLAP = Bounds(64, 512, 128, 32)
USE_TILE_VAE_DEFAULT = True

# The longest output edge is capped at this many times the shortest one: a very thin input (1x4096 through the API)
# would otherwise ask for a gigapixel output. Up to 4:1 panoramas keep the requested resolution.
MAX_LONG_EDGE_FACTOR = 4
# The scale factor and long edge modes never give a longer output edge than this: the DiT sees the whole image, and
# 6K / 8K outputs would run for a very long time or run out of memory on an 8 GB GPU. The shortest edge mode keeps its
# own range (up to 3840 px on the shortest edge) as before.
MAX_OUTPUT_LONG_EDGE = 3840


class SizeMode(str, Enum):
    """How the output size is chosen; the values are the UI's radio labels and the API's strings, in the UI's order."""

    SCALE_FACTOR = "Scale factor"
    SHORTEST_EDGE = "Shortest edge"
    LONG_EDGE = "Long edge"


# The mode the UI starts in.
DEFAULT_SIZE_MODE = SizeMode.SCALE_FACTOR
# The mode of an API call that sends no mode or an unknown one: callers written before the modes existed send a
# shortest edge in px and must keep getting it.
FALLBACK_SIZE_MODE = SizeMode.SHORTEST_EDGE


@dataclass(frozen=True)
class OutputTarget:
    """What the pipeline gets for one input size: the shortest output edge and the cap on the longest one."""

    resolution: int
    max_resolution: int
    raised_to_minimum: bool  # the requested size was under the 512 px minimum of the shortest edge

    def output_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        """The (width, height) the pipeline gives an input of `size`: the shortest edge resized to `resolution`
        (torchvision truncates the other edge), the whole image scaled down when the longest edge exceeds
        `max_resolution`, both sides rounded down to even numbers."""
        width, height = (max(1, int(side)) for side in size)
        shortest, longest = min(width, height), max(width, height)
        new_short, new_long = self.resolution, int(self.resolution * longest / shortest)
        if new_long > self.max_resolution:
            scale = self.max_resolution / new_long
            new_short, new_long = round(new_short * scale), round(new_long * scale)
        new_short, new_long = new_short // 2 * 2, new_long // 2 * 2
        return (new_short, new_long) if width <= height else (new_long, new_short)

    def output_pixels(self, size: Tuple[int, int]) -> int:
        """Pixels of the upscaled image of an input of `size`."""
        width, height = self.output_size(size)
        return width * height


@dataclass(frozen=True)
class UpscaleSettings:
    """What the pipeline gets besides the models and the seed."""

    resolution: int
    input_noise: float
    latent_noise: float
    use_tile_vae: bool
    tile_size: int
    tile_overlap: int
    size_mode: SizeMode = FALLBACK_SIZE_MODE
    scale_factor: float = SCALE_FACTOR.default
    long_edge: int = int(LONG_EDGE.default)

    def target_for(self, size: Tuple[int, int]) -> OutputTarget:
        """The output size of an input of `size` (width, height) in the chosen mode.

        Shortest edge: `resolution` for every input. Scale factor and long edge: the input's own shortest edge times
        the factor, or the shortest edge that gives the chosen long edge, never past `MAX_OUTPUT_LONG_EDGE` on the long
        edge; a result under 512 px is raised to 512 (the model's smallest size here), so ×1 of a small input is
        an upscale."""
        if self.size_mode is SizeMode.SHORTEST_EDGE:
            return OutputTarget(self.resolution, self.resolution * MAX_LONG_EDGE_FACTOR, False)
        width, height = (max(1, int(side)) for side in size)
        shortest, longest = min(width, height), max(width, height)
        if self.size_mode is SizeMode.SCALE_FACTOR:
            long_edge = min(self.scale_factor * longest, MAX_OUTPUT_LONG_EDGE)
        else:
            long_edge = self.long_edge
        requested = round(long_edge * shortest / longest)
        resolution = int(min(max(requested, RESOLUTION.minimum), RESOLUTION.maximum))
        max_resolution = min(resolution * MAX_LONG_EDGE_FACTOR, MAX_OUTPUT_LONG_EDGE)
        return OutputTarget(resolution, max_resolution, requested < RESOLUTION.minimum)


def parse_upscale_settings(resolution: Any, input_noise: Any, latent_noise: Any, use_tile_vae: Any, tile_size: Any,
                           tile_overlap: Any, size_mode: Any = None, scale_factor: Any = None,
                           long_edge: Any = None) -> UpscaleSettings:
    """Raw values → settings. Out-of-range values are clamped to the slider's range; unreadable ones (text, NaN,
    None) take the slider's default, a missing or unknown size mode `FALLBACK_SIZE_MODE`. The tile overlap is also capped at half
    the tile size."""
    tile = int(clamp_to_bounds(tile_size, TILE_SIZE))
    return UpscaleSettings(
        resolution=int(clamp_to_bounds(resolution, RESOLUTION)),
        input_noise=clamp_to_bounds(input_noise, NOISE),
        latent_noise=clamp_to_bounds(latent_noise, NOISE),
        use_tile_vae=USE_TILE_VAE_DEFAULT if use_tile_vae is None else parse_flag(use_tile_vae),
        tile_size=tile,
        # The VAE steps by tile minus overlap: an overlap near the tile size (the overlap slider allows 512 next to a
        # 512 tile) means thousands of tiles and a phase that looks frozen. Half the tile keeps it to a few times the tiles.
        tile_overlap=min(int(clamp_to_bounds(tile_overlap, TILE_OVERLAP)), tile // 2),
        size_mode=parse_size_mode(size_mode),
        scale_factor=round(clamp_to_bounds(scale_factor, SCALE_FACTOR), 2),
        long_edge=int(clamp_to_bounds(long_edge, LONG_EDGE)),
    )


def parse_size_mode(value: Any) -> SizeMode:
    """The size mode named by `value` (case and surrounding spaces ignored), or `FALLBACK_SIZE_MODE`."""
    if isinstance(value, SizeMode):
        return value
    if isinstance(value, str):
        wanted = value.strip().casefold()
        for mode in SizeMode:
            if mode.value.casefold() == wanted:
                return mode
    return FALLBACK_SIZE_MODE


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

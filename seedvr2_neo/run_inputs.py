"""Which images one SeedVR2 run upscales, and in what form the pipeline gets them.

The script (`scripts/seedvr2_script.py`) passes Forge's lists and PIL images in; this module needs only Pillow, so it
is tested outside Forge. What happens to the results (saving, infotexts): `seedvr2_neo/run_results.py`.
"""

import random
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image

T = TypeVar("T")

# Upscaled pixels one pipeline pass may hold: 16 frames of 1024x1024, 4 of 2048x2048, 1 of 4K. A typical batch of 8 at
# 1080p fits in one pass; larger ones are split, and with "Unload" the models reload for each extra pass.
PASS_BUDGET_PIXELS = 16 * 1024 * 1024

# The seed value meaning "not chosen": follow the base image's seed, or pick a random one.
RANDOM_SEED = -1
# Forge's own seeds go up to 2**32 - 1; a random SeedVR2 seed stays in the int32 range it always used.
MAX_SEED = 2 ** 32 - 1
MAX_RANDOM_SEED = 2147483647

# The pipeline takes 3-channel images, or 4-channel ones through its alpha path.
RGB_MODE = "RGB"
RGBA_MODE = "RGBA"
# Modes whose pixels carry alpha; everything else is converted to RGB.
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
_PREMULTIPLIED_GRAYSCALE_MODE = "La"
# 16-bit grayscale (PNG) opens as one of these; values go up to 65535.
_WIDE_GRAYSCALE_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})
_WIDE_TO_8_BIT = 255 / 65535


def select_generated_images(images: Sequence[T], index_of_first_image: Any, generated_count: Any) -> List[T]:
    """The generated images of a `process_images` result, without the grid in front of them.

    Forge puts the grid at index 0 and sets `index_of_first_image` to 1; anything after the generated images is not
    ours to upscale. Out-of-range or unreadable values give fewer images, never an error.
    """
    first = _to_non_negative_int(index_of_first_image)
    count = _to_non_negative_int(generated_count)
    return list(images[first:first + count])


def select_distinct_images(images: Sequence[T]) -> List[T]:
    """Each input object once, in order. The img2img Batch tab passes one file as `[image] * batch_size`; upscaling
    the same object again would give the same picture."""
    seen = set()
    distinct = []
    for image in images:
        if id(image) not in seen:
            seen.add(id(image))
            distinct.append(image)
    return distinct


def to_pipeline_image(image: Image.Image) -> Image.Image:
    """The image as RGB, or RGBA when it has transparency, which is all the pipeline accepts.

    Grayscale, palette, CMYK and other modes would reach the pipeline with the wrong number of channels. The result is
    also what comes back when the upscale fails, so it must be writable as PNG (CMYK is not).
    """
    if image.mode == RGBA_MODE:
        return image
    if image.mode in _WIDE_GRAYSCALE_MODES:
        # Checked before transparency: a 16-bit transparency key would send the image down the alpha path, where
        # Pillow clips the 16-bit values at 255 and the picture turns white. The key itself is dropped.
        return _wide_grayscale_to_8_bit(image).convert(RGB_MODE)
    if image.mode in _ALPHA_MODES or image.info.get("transparency") is not None:
        if image.mode == _PREMULTIPLIED_GRAYSCALE_MODE:
            image = image.convert("LA")  # Pillow has no direct La -> RGBA conversion
        return image.convert(RGBA_MODE)
    if image.mode == RGB_MODE:
        return image
    return image.convert(RGB_MODE)


def group_by_shape(images: Sequence[Image.Image]) -> List[List[int]]:
    """Indices of the images grouped by size and mode, groups in order of first appearance.

    Images of one group can share one tensor, so the models load once for them; another size or channel count cannot.
    """
    groups: Dict[Hashable, List[int]] = {}
    for index, image in enumerate(images):
        groups.setdefault((image.size, image.mode), []).append(index)
    return list(groups.values())


def split_into_passes(groups: Sequence[Sequence[int]], images: Sequence[Image.Image], resolution: int,
                      budget_pixels: int = PASS_BUDGET_PIXELS) -> List[List[int]]:
    """The groups cut into pipeline passes whose upscaled frames add up to at most `budget_pixels`.

    A pass holds all its input and output frames in RAM at once; without a limit a batch of hundreds of images, or an
    API call with many inputs, would exhaust it. One frame larger than the budget still gets a pass of its own.
    """
    passes: List[List[int]] = []
    for group in groups:
        current: List[int] = []
        current_pixels = 0
        for index in group:
            pixels = estimate_output_pixels(images[index].size, resolution)
            if current and current_pixels + pixels > budget_pixels:
                passes.append(current)
                current, current_pixels = [], 0
            current.append(index)
            current_pixels += pixels
        if current:
            passes.append(current)
    return passes


def estimate_output_pixels(size: Tuple[int, int], resolution: int) -> int:
    """Pixels of an image of `size` (width, height) after upscaling its shortest edge to `resolution`."""
    width, height = (max(1, int(side)) for side in size)
    shortest, longest = min(width, height), max(width, height)
    return int(resolution * resolution * longest / shortest)


def _wide_grayscale_to_8_bit(image: Image.Image) -> Image.Image:
    # Through numpy: Pillow's own conversions of 16-bit modes clip at 255 (I;16N even before any scaling).
    values = np.asarray(image, dtype=np.float64) * _WIDE_TO_8_BIT
    return Image.fromarray(np.clip(np.rint(values), 0, 255).astype(np.uint8), mode="L")


def resolve_seed(ui_seed: Any, image_seeds: Optional[Sequence[Any]], base_seed: Any,
                 rng: Optional[random.Random] = None) -> int:
    """The SeedVR2 seed of a run: the one typed in the UI, else the first image's seed, else the run's seed, else
    random. A value an API caller sent that is not a number counts as "not chosen"."""
    for candidate in (ui_seed, _first_or_none(image_seeds), base_seed):
        seed = parse_seed(candidate)
        if seed != RANDOM_SEED:
            return seed
    return (rng or random).randint(0, MAX_RANDOM_SEED)


def is_unusable_seed(value: Any) -> bool:
    """True for a seed that was given (not -1 / empty) but cannot be used, so the caller can say it is replaced."""
    if value is None or isinstance(value, bool):
        return False
    try:
        if float(value) == RANDOM_SEED:
            return False
    except (TypeError, ValueError, OverflowError):
        return True
    return parse_seed(value) == RANDOM_SEED


def parse_seed(value: Any) -> int:
    """A seed the pipeline accepts (0 … 2**32 - 1: it also seeds numpy), or RANDOM_SEED for anything else."""
    if isinstance(value, bool):
        return RANDOM_SEED
    try:
        seed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return RANDOM_SEED
    return seed if 0 <= seed <= MAX_SEED else RANDOM_SEED


def _first_or_none(values: Optional[Sequence[Any]]) -> Any:
    try:
        return values[0] if values else None
    except (TypeError, IndexError, KeyError):
        return None


def _to_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0

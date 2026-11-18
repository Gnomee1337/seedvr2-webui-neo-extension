"""Which images one SeedVR2 run upscales, and in what form the pipeline gets them.

The script (`scripts/seedvr2_script.py`) passes Forge's lists and PIL images in; this module needs only Pillow, so it
is tested outside Forge. What happens to the results (saving, infotexts): `seedvr2_neo/run_results.py`.
"""

import random
from typing import Any, Dict, Hashable, List, Optional, Sequence, TypeVar

from PIL import Image

T = TypeVar("T")

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

    Grayscale, palette, CMYK and other modes would reach the pipeline with the wrong number of channels.
    """
    if image.mode in (RGB_MODE, RGBA_MODE):
        return image
    if image.mode in _ALPHA_MODES or "transparency" in image.info:
        if image.mode == _PREMULTIPLIED_GRAYSCALE_MODE:
            image = image.convert("LA")  # Pillow has no direct La -> RGBA conversion
        return image.convert(RGBA_MODE)
    if image.mode in _WIDE_GRAYSCALE_MODES:
        # A plain convert() clips 16-bit values at 255, turning most of the picture white.
        return image.convert("I").point(lambda value: value * _WIDE_TO_8_BIT).convert("L").convert(RGB_MODE)
    return image.convert(RGB_MODE)


def group_by_shape(images: Sequence[Image.Image]) -> List[List[int]]:
    """Indices of the images grouped by size and mode, groups in order of first appearance.

    One group goes through the pipeline in one pass, as frames of one tensor, so the models load once for it.
    Images of another size or channel count cannot share that tensor.
    """
    groups: Dict[Hashable, List[int]] = {}
    for index, image in enumerate(images):
        groups.setdefault((image.size, image.mode), []).append(index)
    return list(groups.values())


def resolve_seed(ui_seed: Any, image_seeds: Optional[Sequence[Any]], base_seed: Any,
                 rng: Optional[random.Random] = None) -> int:
    """The SeedVR2 seed of a run: the one typed in the UI, else the first image's seed, else the run's seed, else
    random. A value an API caller sent that is not a number counts as "not chosen"."""
    for candidate in (ui_seed, _first_or_none(image_seeds), base_seed):
        seed = _parse_seed(candidate)
        if seed != RANDOM_SEED:
            return seed
    return (rng or random).randint(0, MAX_RANDOM_SEED)


def _parse_seed(value: Any) -> int:
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

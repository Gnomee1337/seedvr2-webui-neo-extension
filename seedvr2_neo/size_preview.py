"""The output size shown next to the size controls before a run, from the settings and the size of the source image.

The script (`scripts/seedvr2_script.py`) passes in the raw values of the size controls and the source size it can read
from Forge's UI; the size itself comes from the same `UpscaleSettings.target_for` the run uses, so the preview and the
result agree. No Forge or Gradio imports.
"""

import math
from typing import Any, Optional, Tuple

from seedvr2_neo.upscale_settings import parse_upscale_settings

# Forge's default `res_step` (Settings: the step of the width / height sliders), which Hires. fix rounds its size to.
DEFAULT_RES_STEP = 64

# What the preview says when the source size is not known before the run (img2img with no image yet, the Batch tab).
UNKNOWN_SOURCE_NOTE = "Output size: depends on each input image"


def describe_output_size(source_size: Optional[Tuple[int, int]], resolution: Any, size_mode: Any, scale_factor: Any,
                         long_edge: Any) -> str:
    """One Markdown line naming the output size of an image of `source_size` (width, height) with these size controls,
    or `UNKNOWN_SOURCE_NOTE` when the size is unknown. Raw values are read like the run reads them."""
    if not _is_usable_size(source_size):
        return UNKNOWN_SOURCE_NOTE
    width, height = (int(side) for side in source_size)
    settings = parse_upscale_settings(resolution, None, None, None, None, None, size_mode, scale_factor, long_edge)
    try:
        target = settings.target_for((width, height))
        out_width, out_height = target.output_size((width, height))
    except OverflowError:  # only a hand-made request sends a side this large
        return UNKNOWN_SOURCE_NOTE
    if not (out_width and out_height):
        return f"Output size: {width}×{height} is too thin to upscale (a side would be 0 px)"
    return f"Output size: **{out_width}×{out_height}** from {width}×{height}{target.raise_note()}"


def _is_usable_size(size: Any) -> bool:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        return False
    try:
        return all(int(side) > 0 for side in size)
    except (TypeError, ValueError, OverflowError):
        return False


def txt2img_final_size(width: Any, height: Any, enable_hr: Any, hr_scale: Any, hr_resize_x: Any, hr_resize_y: Any,
                       res_step: Any = DEFAULT_RES_STEP) -> Optional[Tuple[int, int]]:
    """The size of the image txt2img gives the upscale: width x height, or the Hires. fix size when it is on.

    The Hires. fix size follows Forge's `calculate_target_resolution` (`modules/processing.py`): "Upscale by" when both
    "Resize width/height to" are 0; one of them set keeps the aspect ratio; both set are taken as they are; the result
    is rounded to `res_step`. None when a value is not a number."""
    try:
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            return None
        if not enable_hr:
            return width, height
        resize_x, resize_y = int(hr_resize_x or 0), int(hr_resize_y or 0)
        step = _parse_step(res_step)
        if resize_x == 0 and resize_y == 0:
            scale = float(hr_scale)
            to_x, to_y = width * scale, height * scale
        elif resize_y == 0:
            to_x, to_y = resize_x, resize_x * height / width
        elif resize_x == 0:
            to_x, to_y = resize_y * width / height, resize_y
        else:
            to_x, to_y = resize_x, resize_y
        size = (_round_to_step(to_x, step), _round_to_step(to_y, step))
    except (TypeError, ValueError, OverflowError):
        return None
    return size if all(side > 0 for side in size) else None


def _round_to_step(value: float, step: int) -> int:
    """Forge's `sRound`: to the nearest multiple of `step`, halves up."""
    return math.floor(value / step + 0.5) * step


def _parse_step(res_step: Any) -> int:
    """`res_step` as a positive int, or Forge's default when it is not one."""
    try:
        step = int(res_step)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_RES_STEP
    return step if step > 0 else DEFAULT_RES_STEP

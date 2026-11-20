"""The pipeline's non-numeric settings — color correction and the attention backend — read from raw UI / API values.

An API caller that sends neither gets the values every run used before they were selectable (LAB, `sdpa`). Which
attention backends exist on this machine is known in two ways: the UI lists the ones whose packages are installed
(`installed_attention_modes`, no import, so building the UI loads nothing), and a run checks the vendored pipeline's
own import results (`resolve_attention_mode`), since an installed package can still fail to import. No Forge imports.
"""

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar


class ColorCorrection(str, Enum):
    """The vendored postprocessing's color correction methods; the values are the UI's choices and the API's strings."""

    LAB = "lab"
    WAVELET = "wavelet"
    WAVELET_ADAPTIVE = "wavelet_adaptive"
    HSV = "hsv"
    ADAIN = "adain"
    NONE = "none"


class AttentionMode(str, Enum):
    """The vendored DiT's attention backends; the values are the UI's choices and the API's strings."""

    SDPA = "sdpa"
    FLASH_ATTN_2 = "flash_attn_2"
    FLASH_ATTN_3 = "flash_attn_3"
    SAGEATTN_2 = "sageattn_2"
    SAGEATTN_3 = "sageattn_3"


# LAB keeps the input's colors; `sdpa` is PyTorch's own attention and needs no extra package.
DEFAULT_COLOR_CORRECTION = ColorCorrection.LAB
DEFAULT_ATTENTION_MODE = AttentionMode.SDPA

# The top-level package each backend is imported from in `seedvr2_vendor/optimization/compatibility.py`. SageAttention
# 3 can also come from a recent `sageattention`, but only on RTX 50xx; the UI lists it for its own package only, and an
# API caller can still ask for it (the run checks the real import).
ATTENTION_PACKAGES = {
    AttentionMode.FLASH_ATTN_2: "flash_attn",
    AttentionMode.FLASH_ATTN_3: "flash_attn_interface",
    AttentionMode.SAGEATTN_2: "sageattention",
    AttentionMode.SAGEATTN_3: "sageattn3",
}

E = TypeVar("E", bound=Enum)


@dataclass(frozen=True)
class PipelineOptions:
    """What the pipeline gets besides the models, the seed and the numeric settings."""

    color_correction: ColorCorrection = DEFAULT_COLOR_CORRECTION
    attention_mode: AttentionMode = DEFAULT_ATTENTION_MODE


def parse_color_correction(value: Any) -> ColorCorrection:
    """The method named by `value` (case and surrounding spaces ignored), or LAB."""
    return _parse_choice(value, ColorCorrection, DEFAULT_COLOR_CORRECTION)


def parse_attention_mode(value: Any) -> AttentionMode:
    """The backend named by `value` (case and surrounding spaces ignored), or `sdpa`."""
    return _parse_choice(value, AttentionMode, DEFAULT_ATTENTION_MODE)


def installed_attention_modes(find_spec: Optional[Callable[[str], Any]] = None) -> List[AttentionMode]:
    """`sdpa` and every backend whose package is installed, in the enum's order (`find_spec`: importlib's by
    default). A broken install still counts here; the run falls back to `sdpa` for it (`resolve_attention_mode`)."""
    find_spec = find_spec or importlib.util.find_spec
    return [DEFAULT_ATTENTION_MODE] + [mode for mode, package in ATTENTION_PACKAGES.items()
                                       if _is_installed(package, find_spec)]


def resolve_attention_mode(requested: AttentionMode,
                           available: Mapping[AttentionMode, bool]) -> Tuple[AttentionMode, Optional[str]]:
    """The backend the run uses and, when it is not the requested one, a console note. `available` is the vendored
    pipeline's own import result per backend; a backend missing from it counts as unavailable."""
    if requested is DEFAULT_ATTENTION_MODE or available.get(requested, False):
        return requested, None
    return DEFAULT_ATTENTION_MODE, (f"Attention '{requested.value}' is not available (its package is missing or "
                                    f"failed to import); using '{DEFAULT_ATTENTION_MODE.value}'.")


def choice_values(choices: Iterable[Enum]) -> List[str]:
    """The UI strings of enum members."""
    return [choice.value for choice in choices]


def _parse_choice(value: Any, enum: Type[E], default: E) -> E:
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        wanted = value.strip().casefold()
        for member in enum:
            if member.value == wanted:
                return member
    return default


def _is_installed(package: str, find_spec: Callable[[str], Any]) -> bool:
    try:
        return find_spec(package) is not None
    except (ImportError, ValueError):  # a broken parent package or a half-installed one
        return False

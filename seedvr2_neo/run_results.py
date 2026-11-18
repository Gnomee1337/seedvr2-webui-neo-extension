"""What one SeedVR2 run gives back: which images, which of them are saved, which parameters they carry.

Forge needs exactly one infotext per returned image, in the same order (`docs/external-apis/forge-neo.md`, "Results,
infotexts and saving"), so a failed image comes back as its input instead of dropping out. The script
(`scripts/seedvr2_script.py`) passes plain values in; no Forge imports. Which images go in: `seedvr2_neo/run_inputs.py`.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Mapping, Sequence, TypeVar

T = TypeVar("T")


class GenerationParam(str, Enum):
    """Keys the run adds to the image's generation parameters (infotext)."""

    MODEL = "SeedVR2 Model"
    VAE = "SeedVR2 VAE"
    RESOLUTION = "SeedVR2 Resolution"
    SEED = "SeedVR2 Seed"
    ERROR = "SeedVR2 Error"


@dataclass(frozen=True)
class SavePlan:
    """Which files the run writes besides what Forge already saved."""

    save_results: bool
    save_originals: bool


@dataclass(frozen=True)
class RunImage(Generic[T]):
    """One returned image: the upscaled one, or the input when its upscale did not finish."""

    image: T
    is_upscaled: bool


def plan_saving(is_txt2img: bool, samples_allowed: bool, saving_disabled_by_caller: bool,
                save_original_requested: bool) -> SavePlan:
    """Results follow Forge's own rule (`p.save_samples()`: "Always save all generated images", the API's
    `save_images`, interruption). In txt2img Forge already saved the base images by that same rule, so an extra
    original is written only when it did not; a caller that disabled saving (API) gets no files at all."""
    base_saved_by_forge = is_txt2img and samples_allowed
    return SavePlan(
        save_results=samples_allowed,
        save_originals=save_original_requested and not saving_disabled_by_caller and not base_saved_by_forge,
    )


def seedvr2_generation_params(dit_model: str, vae_model: str, resolution: int, seed: int) -> Dict[str, Any]:
    """Parameters of a successful upscale, added to the image's infotext."""
    return {
        GenerationParam.MODEL.value: dit_model,
        GenerationParam.VAE.value: vae_model,
        GenerationParam.RESOLUTION.value: resolution,
        GenerationParam.SEED.value: seed,
    }


def failure_generation_params(message: str) -> Dict[str, Any]:
    """The parameter that marks an image returned without upscaling, and why."""
    return {GenerationParam.ERROR.value: message}


def merge_run_images(inputs: Sequence[T], upscaled: Mapping[int, T]) -> List[RunImage[T]]:
    """One entry per input, in input order: the upscaled image where there is one, else the input itself."""
    return [RunImage(upscaled[index], True) if index in upscaled else RunImage(image, False)
            for index, image in enumerate(inputs)]


def append_generation_params(infotext: str, params: Mapping[str, Any]) -> str:
    """`infotext` with `params` added to its last line, where Forge keeps the `key: value` parameters."""
    if not params:
        return infotext
    formatted = format_generation_params(params)
    return f"{infotext}, {formatted}" if infotext else formatted


def format_generation_params(params: Mapping[str, Any]) -> str:
    """`key: value, key: value` in Forge's infotext style, for when Forge's own `create_infotext` cannot be used.
    A value with a comma, colon or line break is quoted, as Forge does, so the line still parses back."""
    return ", ".join(f"{key}: {_quote(value)}" for key, value in params.items())


def _quote(value: Any) -> str:
    text = str(value)
    if any(mark in text for mark in (",", ":", "\n", '"')):
        return json.dumps(text, ensure_ascii=False)
    return text

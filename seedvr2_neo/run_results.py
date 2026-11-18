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

NEGATIVE_PROMPT_LABEL = "Negative prompt: "


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


def failure_generation_params(params: Mapping[str, Any], message: str) -> Dict[str, Any]:
    """`params` of the attempted upscale plus the reason the image came back without it. The SeedVR2 keys stay: Forge
    reads the last infotext line as parameters only when it has at least three of them."""
    return {**params, GenerationParam.ERROR.value: message}


def merge_run_images(inputs: Sequence[T], upscaled: Mapping[int, T]) -> List[RunImage[T]]:
    """One entry per input, in input order: the upscaled image where there is one, else the input itself."""
    return [RunImage(upscaled[index], True) if index in upscaled else RunImage(image, False)
            for index, image in enumerate(inputs)]


@dataclass(frozen=True)
class BaseInfotext:
    """An infotext split into the prompt part and the `key: value` parameters line Forge keeps last.

    The SeedVR2 parameters are added to that line instead of rebuilding the infotext with Forge's `create_infotext`:
    it reads the SD model's fields, and the checkpoint is usually unloaded (or never loaded) by the time the upscale
    ends."""

    prompt_part: str
    params_line: str

    @classmethod
    def from_forge(cls, infotext: str) -> "BaseInfotext":
        """An infotext Forge built for a generated image; its last line is the parameters."""
        prompt_part, _, params_line = (infotext or "").rpartition("\n")
        return cls(prompt_part, params_line)

    @classmethod
    def from_prompt(cls, prompt: str, negative_prompt: str) -> "BaseInfotext":
        """For an image Forge did not generate (img2img input): the prompt in Forge's layout, no parameters yet."""
        negative = f"\n{NEGATIVE_PROMPT_LABEL}{negative_prompt}" if negative_prompt else ""
        return cls(f"{prompt or ''}{negative}".strip(), "")

    def with_params(self, params: Mapping[str, Any]) -> str:
        """The full infotext with `params` appended to the parameters line."""
        params_line = ", ".join(part for part in (self.params_line, format_generation_params(params)) if part)
        return "\n".join(part for part in (self.prompt_part, params_line) if part)


@dataclass(frozen=True)
class InputDetails:
    """What a returned image takes from its input: seed and prompt for the file name, the infotext to extend."""

    seed: int
    prompt: str
    infotext: BaseInfotext


def format_generation_params(params: Mapping[str, Any]) -> str:
    """`key: value, key: value` in Forge's infotext style. A value with a comma, colon, quote or line break is quoted,
    as Forge does, so the line still parses back."""
    return ", ".join(f"{key}: {_quote(value)}" for key, value in params.items())


def _quote(value: Any) -> str:
    text = str(value)
    if any(mark in text for mark in (",", ":", "\n", '"')):
        return json.dumps(text, ensure_ascii=False)
    return text

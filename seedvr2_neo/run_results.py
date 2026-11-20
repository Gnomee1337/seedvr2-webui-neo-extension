"""What one SeedVR2 run gives back: which images, which of them are saved, which parameters they carry.

Forge needs exactly one infotext per returned image, in the same order (`docs/external-apis/forge-neo.md`, "Results,
infotexts and saving"), so a failed image comes back as its input instead of dropping out. The script
(`scripts/seedvr2_script.py`) passes plain values in; no Forge imports. Which images go in: `seedvr2_neo/run_inputs.py`.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

from seedvr2_neo.pipeline_options import PipelineOptions
from seedvr2_neo.upscale_settings import SizeMode, UpscaleSettings

T = TypeVar("T")

NEGATIVE_PROMPT_LABEL = "Negative prompt: "

# Where Forge keeps the infotext inside an image (`image.info`), and why an image came back without upscaling when
# the run gives no other reason.
PNG_INFO_KEY = "parameters"
NOT_UPSCALED_MESSAGE = "not upscaled"


class GenerationParam(str, Enum):
    """Keys the run adds to the image's generation parameters (infotext)."""

    MODEL = "SeedVR2 Model"
    VAE = "SeedVR2 VAE"
    RESOLUTION = "SeedVR2 Resolution"
    SEED = "SeedVR2 Seed"
    SIZE_MODE = "SeedVR2 Size Mode"
    SCALE_FACTOR = "SeedVR2 Scale"
    LONG_EDGE = "SeedVR2 Long Edge"
    COLOR_CORRECTION = "SeedVR2 Color Correction"
    ATTENTION = "SeedVR2 Attention"
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


def plan_saving(is_txt2img: bool, samples_save: bool, saving_disabled_by_caller: bool,
                save_original_requested: bool) -> SavePlan:
    """Results are saved when "Always save all generated images" is on and the caller did not disable saving (the
    API's `save_images`); with saving off the extension writes nothing at all.

    Not `p.save_samples()`: it also reads Forge's interrupted / skipped state, and a Skip on the last txt2img batch
    stays set after the base images, which would stop saving for the whole run. The extra original is written only
    for img2img inputs: in txt2img Forge itself saved the base images by the same rule."""
    save_results = samples_save and not saving_disabled_by_caller
    return SavePlan(
        save_results=save_results,
        save_originals=save_original_requested and save_results and not is_txt2img,
    )


def read_option(override_settings: Any, name: str, default: Any) -> Any:
    """A Forge option as this run sees it: the request's `override_settings` value when it has one, else `default`.

    Forge applies `override_settings` only inside `process_images` (and restores them afterwards), so the script
    reads them itself: img2img never calls it (the Batch tab turns date subfolders off that way), and in txt2img the
    API's overrides are already restored when the upscaled images are saved."""
    if isinstance(override_settings, Mapping) and name in override_settings:
        return override_settings[name]
    return default


def seedvr2_generation_params(dit_model: str, vae_model: str, settings: UpscaleSettings, size: Tuple[int, int],
                              seed: int, options: PipelineOptions = PipelineOptions()) -> Dict[str, Any]:
    """Parameters of the upscale of an input of `size` (width, height), added to its infotext. The resolution is the
    shortest edge this input was upscaled to, which in the scale factor and long edge modes depends on its size. The
    attention backend is the one the run used, after any fallback to `sdpa`."""
    params = {
        GenerationParam.MODEL.value: dit_model,
        GenerationParam.VAE.value: vae_model,
        GenerationParam.RESOLUTION.value: settings.target_for(size).resolution,
        GenerationParam.SEED.value: seed,
        GenerationParam.SIZE_MODE.value: settings.size_mode.value,
    }
    if settings.size_mode is SizeMode.SCALE_FACTOR:
        params[GenerationParam.SCALE_FACTOR.value] = settings.scale_factor
    elif settings.size_mode is SizeMode.LONG_EDGE:
        params[GenerationParam.LONG_EDGE.value] = settings.long_edge
    params[GenerationParam.COLOR_CORRECTION.value] = options.color_correction.value
    params[GenerationParam.ATTENTION.value] = options.attention_mode.value
    return params


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
        # Trailing whitespace is dropped first: after a final line break the parameters line would look empty, and the
        # SeedVR2 keys would start a new line that Forge reads as the only parameters.
        prompt_part, _, params_line = (infotext or "").rstrip().rpartition("\n")
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


def describe_generated_inputs(seeds: Sequence[int], prompts: Sequence[str], base_infotexts: Sequence[str],
                              count: int) -> List[InputDetails]:
    """Seed, prompt and Forge's own infotext of each txt2img image, in Forge's `all_seeds` order (`count` is never
    more than `len(seeds)`). A missing infotext gives an image with only the SeedVR2 parameters."""
    return [InputDetails(seeds[index], prompts[index],
                         BaseInfotext.from_forge(base_infotexts[index] if index < len(base_infotexts) else ""))
            for index in range(count)]


def describe_uploaded_inputs(seed: int, prompt: str, negative_prompt: str, count: int) -> List[InputDetails]:
    """img2img inputs have no seed or infotext of their own: the SeedVR2 seed and the prompt from the UI."""
    details = InputDetails(seed, prompt, BaseInfotext.from_prompt(prompt, negative_prompt))
    return [details] * count


@dataclass(frozen=True)
class ResultImage:
    """One image of the result as Forge gets it, with its infotext and what its file is named after."""

    image: Any
    infotext: str
    details: InputDetails
    is_upscaled: bool


def finish_images(run_images: Sequence[RunImage], details: Sequence[InputDetails],
                  params: Sequence[Mapping[str, Any]], failure: Optional[str],
                  embed_infotext: bool) -> List[ResultImage]:
    """One image and one infotext per input, in input order (Forge pairs them by index). Upscaled images carry their
    SeedVR2 parameters (`params`, one mapping per input); the others also the reason they were not upscaled. With
    `embed_infotext` (Forge's "Save text information about generation parameters as chunks to png files") the
    infotext also goes into the image, where Forge's gallery and API look for it when the image is not saved to disk."""
    reason = failure or NOT_UPSCALED_MESSAGE
    results = []
    for run_image, input_details, image_params in zip(run_images, details, params, strict=True):
        if not run_image.is_upscaled:
            image_params = failure_generation_params(image_params, reason)
        infotext = input_details.infotext.with_params(image_params)
        # Upscaled images are the run's own; inputs and Forge's base images are copied before their info changes
        image = run_image.image if run_image.is_upscaled else copy_keeping_saved_file(run_image.image)
        if embed_infotext:
            image.info[PNG_INFO_KEY] = infotext
        results.append(ResultImage(image, infotext, input_details, run_image.is_upscaled))
    return results


def not_upscaled_comment(run_images: Sequence[RunImage], failure: Optional[str]) -> Optional[str]:
    """The result comment when some images come back without upscaling, else None."""
    not_upscaled = sum(1 for run_image in run_images if not run_image.is_upscaled)
    if not not_upscaled:
        return None
    return (f"SeedVR2: {failure or NOT_UPSCALED_MESSAGE} — {not_upscaled} of {len(run_images)} images returned "
            f"without upscaling")


def copy_keeping_saved_file(image: Any) -> Any:
    """A copy of a PIL image that still points the gallery to the file Forge already saved (`already_saved_as`), so
    it serves that file instead of writing a temporary one."""
    copied = image.copy()
    saved_as = getattr(image, "already_saved_as", None)
    if saved_as:
        copied.already_saved_as = saved_as
    return copied


def format_generation_params(params: Mapping[str, Any]) -> str:
    """`key: value, key: value` in Forge's infotext style. A value with a comma, colon, quote or line break is quoted,
    as Forge does, so the line still parses back."""
    return ", ".join(f"{key}: {_quote(value)}" for key, value in params.items())


def _quote(value: Any) -> str:
    text = str(value)
    if any(mark in text for mark in (",", ":", "\n", '"')):
        return json.dumps(text, ensure_ascii=False)
    return text

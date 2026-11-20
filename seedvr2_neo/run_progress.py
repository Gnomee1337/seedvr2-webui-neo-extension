"""How far one upscale run is, as a fraction for the host's progress bar plus a short text.

A run is passes of images; each pass goes through the four pipeline phases. The pipeline reports each finished batch
(one image with batch size 1) through its progress callback; inside the DiT phase, where one image can take minutes,
each DiT block that starts moves the bar too. Phases get weights by their rough share of a run's time, so the bar
and Forge's ETA move about evenly. No Forge or torch imports: the script turns the fraction into `shared.state`.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class Phase:
    """One pipeline phase: its console name and its rough share of a run's time."""

    name: str
    weight: float


# The order the pipeline runs them in. Weights: rough shares of a run's time on an 8 GB GPU with VAE tiling; the DiT
# and the tiled decode take most of it.
PHASES = (Phase("Encoding", 1.0), Phase("Upscaling", 4.0), Phase("Decoding", 3.0), Phase("Post-processing", 0.5))
# Indexes into `PHASES`; the DiT phase is the one whose blocks move the bar.
ENCODE_PHASE, UPSCALE_PHASE, DECODE_PHASE, POSTPROCESS_PHASE = range(len(PHASES))

ProgressReport = Callable[[float, str], None]


class RunProgress:
    """Counts what is done and reports the fraction of the whole run (0…1) with a text like
    "SeedVR2: Upscaling, pass 1 of 2" after every change."""

    def __init__(self, pass_sizes: Sequence[int], report: ProgressReport) -> None:
        """`pass_sizes`: how many images each pass upscales, in order."""
        self._pass_sizes = [max(1, int(size)) for size in pass_sizes] or [1]
        self._total_images = sum(self._pass_sizes)
        self._report = report
        self._pass = 0
        self._phase = 0
        self._batch_fraction = 0.0  # done share of the current phase, from the pipeline's callback
        self._blocks_per_image: Optional[int] = None
        self._blocks_seen = 0  # DiT blocks started for the current image

    def start_pass(self, index: int) -> None:
        self._pass = min(max(0, index), len(self._pass_sizes) - 1)
        self._phase = 0
        self._reset_phase()
        self._publish()

    def start_phase(self, index: int, blocks_per_image: Optional[int] = None) -> None:
        """`blocks_per_image`: in the DiT phase, how many DiT blocks run per image; None when unknown or not the DiT."""
        self._phase = min(max(0, index), len(PHASES) - 1)
        self._reset_phase()
        self._blocks_per_image = blocks_per_image if blocks_per_image and blocks_per_image > 0 else None
        self._publish()

    def batch_done(self, current: object, total: object) -> None:
        """The pipeline's progress callback: `current` of `total` batches of the phase are done. Values that are not
        finite numbers are ignored: the bar is feedback only and must never fail the upscale."""
        try:
            fraction = float(current) / float(total)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return
        if not math.isfinite(fraction):
            return
        self._batch_fraction = min(max(fraction, self._batch_fraction), 1.0)
        self._blocks_seen = 0
        self._publish()

    def block_started(self) -> None:
        """A DiT block started. Counts only in the DiT phase when the block count is known; the share of the current
        image never passes the image itself (a block counted twice, a guidance pass) before its batch is done."""
        if self._phase != UPSCALE_PHASE or self._blocks_per_image is None:
            return
        self._blocks_seen += 1
        self._publish()

    @property
    def fraction(self) -> float:
        images_before = sum(self._pass_sizes[:self._pass])
        pass_size = self._pass_sizes[self._pass]
        total_weight = sum(phase.weight for phase in PHASES)
        weight_before = sum(phase.weight for phase in PHASES[:self._phase])
        phase_share = self._batch_fraction
        if self._blocks_per_image is not None and self._blocks_seen:
            image_share = min(self._blocks_seen / self._blocks_per_image, 1.0) / pass_size
            phase_share = min(phase_share + image_share, 1.0)
        pass_fraction = (weight_before + PHASES[self._phase].weight * phase_share) / total_weight
        return min((images_before + pass_fraction * pass_size) / self._total_images, 1.0)

    def describe(self) -> str:
        phase = PHASES[self._phase].name
        if len(self._pass_sizes) == 1:
            return f"SeedVR2: {phase}"
        return f"SeedVR2: {phase}, pass {self._pass + 1} of {len(self._pass_sizes)}"

    def _reset_phase(self) -> None:
        self._batch_fraction = 0.0
        self._blocks_per_image = None
        self._blocks_seen = 0

    def _publish(self) -> None:
        self._report(self.fraction, self.describe())

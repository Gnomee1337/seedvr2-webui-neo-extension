"""GPU profiles: starting values of the memory sliders (VAE tile size and overlap, BlockSwap) for a class of GPU.

Choosing a profile only fills the sliders; the sliders stay visible and editable, and the profile itself is not a
script argument, so an API call and the infotext never depend on it. BlockSwap depends on the DiT size too: 7B needs
blocks in RAM where 3B fits. No Forge imports.
"""

from dataclasses import dataclass
from typing import List, Optional

from seedvr2_neo.memory_plan import RECOMMENDED_7B_BLOCKS_TO_SWAP
from seedvr2_neo.upscale_settings import TILE_OVERLAP, TILE_SIZE

# The choice that leaves the sliders as they are.
MANUAL = "Manual"


@dataclass(frozen=True)
class GpuProfile:
    """Slider values for one class of GPU. `measured`: the values were checked in a live run on such a GPU."""

    label: str
    tile_size: int
    tile_overlap: int
    blocks_to_swap_3b: int
    blocks_to_swap_7b: int
    measured: bool

    def blocks_to_swap(self, is_7b: bool) -> int:
        return self.blocks_to_swap_7b if is_7b else self.blocks_to_swap_3b


# 8 GB: the defaults measured live (tile 512; 3B without BlockSwap; 7B fp8 with 16 blocks). 12 and 24 GB are not
# measured: larger tiles mean fewer of them and a faster VAE; 7B on 12 GB keeps a few blocks in RAM as a safe start
# for outputs above 1080p.
GPU_PROFILES = (
    GpuProfile("8 GB", int(TILE_SIZE.default), int(TILE_OVERLAP.default), 0, RECOMMENDED_7B_BLOCKS_TO_SWAP, True),
    GpuProfile("12 GB", 768, int(TILE_OVERLAP.default), 0, 8, False),
    GpuProfile("24 GB", 1024, int(TILE_OVERLAP.default), 0, 0, False),
)


def profile_choices() -> List[str]:
    """The dropdown's choices: Manual first, then the profiles from the smallest GPU."""
    return [MANUAL] + [profile.label for profile in GPU_PROFILES]


def find_profile(label: object) -> Optional[GpuProfile]:
    """The profile named `label`, or None for Manual and anything unknown."""
    for profile in GPU_PROFILES:
        if profile.label == label:
            return profile
    return None


def describe_profiles() -> str:
    """The dropdown's hint: what a profile fills and which values were not measured."""
    unmeasured = [profile.label for profile in GPU_PROFILES if not profile.measured]
    note = f"; {' and '.join(unmeasured)} values are not measured yet" if unmeasured else ""
    return f"Fills BlockSwap and the VAE tile size for your GPU; the sliders stay editable{note}"

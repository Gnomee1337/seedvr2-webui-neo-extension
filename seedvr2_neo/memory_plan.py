"""Memory settings of one SeedVR2 run, built from the UI values.

`build_memory_plan` turns the "After run" choice and the BlockSwap controls into the
offload devices and cache flags the vendored pipeline expects (`seedvr2_vendor/core/generation_utils.py`,
`seedvr2_vendor/optimization/blockswap.py`). `low_vram_warning` gives the note shown when a 7B DiT is selected on a
small GPU. No Forge, Gradio or torch imports (the vendored `dit_size` reads only file headers): the script
(`scripts/seedvr2_script.py`) passes plain values in.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from seedvr2_vendor.utils.dit_size import DitSize, choose_dit_size

# The vendored pipeline offloads to the CPU by device name; it accepts a string as well as a torch.device.
CPU_DEVICE = "cpu"

# The 7B DiT has 36 transformer blocks, the 3B has 32; the pipeline clamps a larger number to the model's block count.
MAX_BLOCKS_TO_SWAP = 36

# 7B fp8 weights alone take about 7.9 GB; activations at 1080p come on top. GPUs reporting less than this
# (8, 10 and 11 GB cards) spill into shared memory without BlockSwap.
LOW_VRAM_7B_THRESHOLD_GIB = 11.0

_BYTES_PER_GIB = 1024 ** 3


class AfterRun(str, Enum):
    """What happens to the SeedVR2 models when a run ends. Values are the UI labels."""

    UNLOAD = "Unload SeedVR2"
    KEEP_IN_RAM = "Keep in RAM"


@dataclass(frozen=True)
class MemoryPlan:
    """Offload devices and cache flags for one run."""

    cache_models: bool
    dit_offload_device: Optional[str]
    vae_offload_device: Optional[str]
    blocks_to_swap: int
    swap_io_components: bool

    def block_swap_config(self) -> Dict[str, Any]:
        """The `block_swap_config` argument of `prepare_runner`."""
        return {
            "blocks_to_swap": self.blocks_to_swap,
            "swap_io_components": self.swap_io_components,
            "offload_device": self.dit_offload_device,
        }


def build_memory_plan(after_run: str, blocks_to_swap: Any, swap_io_components: Any) -> MemoryPlan:
    """Build the plan from raw UI values.

    An unknown "After run" value falls back to unloading: freeing memory is the safe side. BlockSwap needs a DiT offload
    device different from the GPU (upstream raises otherwise), so the CPU is set whenever BlockSwap is on. BlockSwap means
    VRAM is short, so the VAE also waits in RAM while the DiT runs instead of holding VRAM it does not use then.
    """
    blocks = clamp_blocks_to_swap(blocks_to_swap)
    swap_io = bool(swap_io_components)
    offload_device = CPU_DEVICE if blocks > 0 or swap_io else None
    return MemoryPlan(
        cache_models=_parse_after_run(after_run) is AfterRun.KEEP_IN_RAM,
        dit_offload_device=offload_device,
        vae_offload_device=offload_device,
        blocks_to_swap=blocks,
        swap_io_components=swap_io,
    )


def clamp_blocks_to_swap(value: Any) -> int:
    """Turn a slider value into a block count in [0, MAX_BLOCKS_TO_SWAP]; anything unreadable means BlockSwap off."""
    try:
        blocks = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(blocks, MAX_BLOCKS_TO_SWAP))


def is_7b_checkpoint(dit_model: str, checkpoint_path: str) -> bool:
    """True for a 7B DiT, judged the same way the pipeline picks its config (header first, then the file name)."""
    return choose_dit_size(dit_model, checkpoint_path).size is DitSize.B7


def low_vram_warning(is_7b: bool, total_vram_bytes: Optional[int]) -> Optional[str]:
    """The note for a 7B DiT on a GPU below the threshold, or None.

    The caller tells the size from the checkpoint (`seedvr2_vendor/utils/dit_size.py`); an unknown GPU size gives no
    note, since a wrong warning is worse than none.
    """
    if not is_7b or not total_vram_bytes or total_vram_bytes <= 0:
        return None
    total_gib = total_vram_bytes / _BYTES_PER_GIB
    if total_gib >= LOW_VRAM_7B_THRESHOLD_GIB:
        return None
    return (f"⚠️ 7B model on a {total_gib:.1f} GB GPU: without BlockSwap it spills into shared memory and runs many times "
            f"slower. Turn on BlockSwap below (more blocks = less VRAM, slower), or use a 3B model.")


def _parse_after_run(value: str) -> AfterRun:
    try:
        return AfterRun(value)
    except ValueError:
        return AfterRun.UNLOAD

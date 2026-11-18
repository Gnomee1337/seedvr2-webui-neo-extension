"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

# Local patch (Forge Neo): the vendored package is named `seedvr2_vendor` here, not `src`
from .seedvr2_vendor.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .seedvr2_vendor.interfaces import comfy_entrypoint, SeedVR2Extension

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]
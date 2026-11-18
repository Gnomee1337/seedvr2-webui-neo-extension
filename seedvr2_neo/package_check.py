"""Detect another extension's top-level `src` package loaded instead of our vendored one.

The vendored SeedVR2 pipeline is imported as `src`. Other extensions (the original SeedVR2
port among them) ship a package with the same name, and Python keeps whichever was imported
first. Their copy lacks our local patches, so the run would fail deep inside with
"No module named ..." instead of a clear message.
"""
from __future__ import annotations

import os
from types import ModuleType
from typing import Optional

VENDORED_PACKAGE = "src"


def describe_shadowed_package(package: ModuleType, extension_root: str) -> str:
    """Error text when the imported `package` does not live in `<extension_root>/src`, else ""."""
    loaded = _package_folder(package)
    if loaded and _same_folder(loaded, os.path.join(extension_root, VENDORED_PACKAGE)):
        return ""
    return (
        f"Another extension's '{VENDORED_PACKAGE}' package ({loaded or 'an unknown location'}) is loaded "
        f"instead of SeedVR2's. Disable that extension and restart the WebUI."
    )


def _package_folder(package: ModuleType) -> Optional[str]:
    """Folder of a regular package (`__file__`) or the first folder of a namespace package (`__path__`)."""
    package_file = getattr(package, "__file__", None)
    if package_file:
        return os.path.dirname(os.path.abspath(package_file))
    return next(iter(getattr(package, "__path__", None) or ()), None)


def _same_folder(first: str, second: str) -> bool:
    """Same folder on disk (follows links); falls back to comparing normalized paths if one does not exist."""
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))

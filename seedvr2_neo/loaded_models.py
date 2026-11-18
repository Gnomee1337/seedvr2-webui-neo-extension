"""Which DiT / VAE files the cached SeedVR2 models came from.

The vendored model cache (`seedvr2_vendor/core/model_cache.py`) matches a cached model by file name only, so a file with
the same name in the other model folder, or a file replaced in place, would reuse the old weights. The script
(`scripts/seedvr2_script.py`) asks `LoadedModelTracker` before each run and drops the cache when the files differ.
No Forge or torch imports.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelFileIdentity:
    """A model file as it was on disk: absolute path, size and modification time."""

    path: str
    size: int
    mtime_ns: int


def read_model_file_identity(path: str) -> Optional[ModelFileIdentity]:
    """Identity of the file at `path`, or None when it cannot be read (then the cache is never trusted)."""
    absolute_path = os.path.abspath(path)
    try:
        stat = os.stat(absolute_path)
    except OSError:
        return None
    return ModelFileIdentity(path=os.path.normcase(absolute_path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


class LoadedModelTracker:
    """Remembers the DiT and VAE files of the cached models between runs."""

    def __init__(self) -> None:
        self._dit: Optional[ModelFileIdentity] = None
        self._vae: Optional[ModelFileIdentity] = None

    def is_cache_valid_for(self, dit_path: str, vae_path: str) -> bool:
        """True only when both files are the ones the cached models were loaded from and are unchanged."""
        if self._dit is None or self._vae is None:
            return False
        return (read_model_file_identity(dit_path) == self._dit
                and read_model_file_identity(vae_path) == self._vae)

    def remember(self, dit_path: str, vae_path: str) -> None:
        """Record the files the models now in the cache were loaded from."""
        self._dit = read_model_file_identity(dit_path)
        self._vae = read_model_file_identity(vae_path)

    def forget(self) -> None:
        """The cache was emptied: nothing is trusted until the next `remember`."""
        self._dit = None
        self._vae = None

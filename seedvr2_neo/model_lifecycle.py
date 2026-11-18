"""When the cached SeedVR2 models are dropped: before a run, after it, on errors.

The vendored pipeline keeps the DiT and VAE in a process-wide cache (`seedvr2_vendor/core/model_cache.py`) under fixed
ids. `ModelLifecycle` decides when that cache is emptied; the script (`scripts/seedvr2_script.py`) gives it the cache
and a function that returns freed memory to the system (`gc` + Forge's empty-cache), so this module needs no Forge or
torch and is tested with stubs. Which files the cached models came from: `seedvr2_neo/loaded_models.py`.
"""

from typing import Any, Callable, Dict, Protocol

from seedvr2_neo.loaded_models import LoadedModelTracker

# Cache ids under which the script stores its models in the vendored cache.
DIT_CACHE_ID = "forge_dit"
VAE_CACHE_ID = "forge_vae"


class ModelCache(Protocol):
    """The part of the vendored `GlobalModelCache` used here."""

    def remove_dit(self, dit_config: Dict[str, Any]) -> bool: ...

    def remove_vae(self, vae_config: Dict[str, Any]) -> bool: ...


class ModelLifecycle:
    """Process-wide: Forge creates one Script per tab, but the vendored cache is shared by both."""

    def __init__(self, tracker: LoadedModelTracker, release_memory: Callable[[], None]) -> None:
        self._tracker = tracker
        self._release_memory = release_memory

    def prepare_run(self, cache: ModelCache, dit_path: str, vae_path: str, force_reload: bool) -> bool:
        """Drop the cached models before a run unless they came from exactly these files.

        Returns True when cached models were dropped. Force Reload always drops them, so the files are read from disk.
        """
        if not force_reload and self._tracker.is_cache_valid_for(dit_path, vae_path):
            return False
        return self.unload(cache)

    def finish_run(self, cache: ModelCache, keep_models: bool, succeeded: bool, dit_path: str, vae_path: str) -> None:
        """Keep the models only after a successful run with "Keep in RAM"; after an error or interrupt they may hold
        partial state or sit on the GPU, so they are dropped. Freed memory is returned to the system either way."""
        if keep_models and succeeded:
            self._tracker.remember(dit_path, vae_path)
            self._release_memory()
            return
        if not self.unload(cache):
            self._release_memory()

    def unload(self, cache: ModelCache) -> bool:
        """Remove both models from the cache and release their memory. Returns True when anything was cached."""
        self._tracker.forget()
        removed_dit = cache.remove_dit({"node_id": DIT_CACHE_ID})
        removed_vae = cache.remove_vae({"node_id": VAE_CACHE_ID})
        if removed_dit or removed_vae:
            self._release_memory()
            return True
        return False

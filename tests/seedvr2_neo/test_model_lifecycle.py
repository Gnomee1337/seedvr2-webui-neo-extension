"""When cached models are dropped, `seedvr2_neo/model_lifecycle.py`."""
from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.model_lifecycle import DIT_CACHE_ID, VAE_CACHE_ID, ModelLifecycle

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"


class FakeCache:
    """Stands in for the vendored GlobalModelCache: remembers which ids are cached and what was removed."""

    def __init__(self, cached=()):
        self.cached = set(cached)
        self.removed = []

    def remove_dit(self, dit_config):
        return self._remove(dit_config["node_id"])

    def remove_vae(self, vae_config):
        return self._remove(vae_config["node_id"])

    def _remove(self, node_id):
        self.removed.append(node_id)
        if node_id in self.cached:
            self.cached.discard(node_id)
            return True
        return False


class Lifecycle:
    """A ModelLifecycle with a counter of memory releases."""

    def __init__(self):
        self.releases = 0
        self.lifecycle = ModelLifecycle(LoadedModelTracker(), self._release)

    def _release(self):
        self.releases += 1

    def run(self, cache, dit, vae, keep_models, succeeded=True, force_reload=False):
        """One whole run: prepare, then finish. Returns (dropped before, kept after)."""
        dropped = self.lifecycle.prepare_run(cache, dit, vae, force_reload)
        kept = self.lifecycle.finish_run(cache, keep_models, succeeded)
        return dropped, kept


def make_models(tmp_path):
    (tmp_path / DIT).write_bytes(b"dit")
    (tmp_path / VAE).write_bytes(b"vae")
    return str(tmp_path / DIT), str(tmp_path / VAE)


def full_cache():
    return FakeCache(cached={DIT_CACHE_ID, VAE_CACHE_ID})


def test_first_run_drops_whatever_is_cached(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), full_cache()

    assert life.lifecycle.prepare_run(cache, dit, vae, force_reload=False) is True
    assert cache.cached == set()
    assert life.releases == 1


def test_kept_models_are_reused_for_the_same_files(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), full_cache()
    assert life.run(cache, dit, vae, keep_models=True)[1] is True
    cache.cached = {DIT_CACHE_ID, VAE_CACHE_ID}  # the pipeline cached them during the run

    assert life.lifecycle.prepare_run(cache, dit, vae, force_reload=False) is False
    assert cache.cached == {DIT_CACHE_ID, VAE_CACHE_ID}


def test_force_reload_drops_kept_models(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), full_cache()
    life.run(cache, dit, vae, keep_models=True)
    cache.cached = {DIT_CACHE_ID, VAE_CACHE_ID}

    assert life.lifecycle.prepare_run(cache, dit, vae, force_reload=True) is True
    assert cache.cached == set()


def test_changed_file_drops_kept_models(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), full_cache()
    life.run(cache, dit, vae, keep_models=True)
    cache.cached = {DIT_CACHE_ID, VAE_CACHE_ID}
    with open(dit, "wb") as f:
        f.write(b"replaced with a bigger file")

    assert life.lifecycle.prepare_run(cache, dit, vae, force_reload=False) is True


def test_file_replaced_during_the_run_is_not_trusted_next_time(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), FakeCache()
    life.lifecycle.prepare_run(cache, dit, vae, force_reload=False)
    with open(dit, "wb") as f:  # replaced while the old weights were being loaded
        f.write(b"replaced with a bigger file")
    life.lifecycle.finish_run(cache, keep_models=True, succeeded=True)

    assert life.lifecycle.prepare_run(full_cache(), dit, vae, force_reload=False) is True


def test_unload_mode_drops_models_after_a_successful_run(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), FakeCache()
    life.lifecycle.prepare_run(cache, dit, vae, force_reload=False)
    cache.cached = {DIT_CACHE_ID, VAE_CACHE_ID}

    assert life.lifecycle.finish_run(cache, keep_models=False, succeeded=True) is False
    assert cache.cached == set()


def test_failed_run_drops_models_even_in_keep_mode(tmp_path):
    dit, vae = make_models(tmp_path)
    life, cache = Lifecycle(), FakeCache()
    life.lifecycle.prepare_run(cache, dit, vae, force_reload=False)
    cache.cached = {DIT_CACHE_ID, VAE_CACHE_ID}

    assert life.lifecycle.finish_run(cache, keep_models=True, succeeded=False) is False
    assert cache.cached == set()
    # and nothing is trusted afterwards
    assert life.lifecycle.prepare_run(full_cache(), dit, vae, force_reload=False) is True


def test_finish_without_prepare_keeps_nothing(tmp_path):
    life, cache = Lifecycle(), full_cache()

    assert life.lifecycle.finish_run(cache, keep_models=True, succeeded=True) is False
    assert cache.cached == set()


def test_memory_is_released_once_per_finish(tmp_path):
    dit, vae = make_models(tmp_path)
    for keep, succeeded, cached in [(True, True, ()), (False, True, ()), (True, False, ()), (False, False, ()),
                                    (False, True, (DIT_CACHE_ID, VAE_CACHE_ID))]:
        life, cache = Lifecycle(), FakeCache()
        life.lifecycle.prepare_run(cache, dit, vae, force_reload=False)
        cache.cached = set(cached)
        life.lifecycle.finish_run(cache, keep_models=keep, succeeded=succeeded)
        assert life.releases == 1


def test_unload_removes_both_ids_and_reports_empty_cache():
    life, cache = Lifecycle(), FakeCache()

    assert life.lifecycle.unload(cache) is False
    assert cache.removed == [DIT_CACHE_ID, VAE_CACHE_ID]
    assert life.releases == 0


def test_partial_cache_is_dropped():
    life, cache = Lifecycle(), FakeCache(cached={VAE_CACHE_ID})

    assert life.lifecycle.unload(cache) is True
    assert cache.cached == set()

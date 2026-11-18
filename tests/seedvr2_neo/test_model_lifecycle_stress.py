"""Hostile call orders, failing caches and changing files for `seedvr2_neo/model_lifecycle.py`.

What must hold whatever happens: cached weights are never reused for files that changed, a failure never leaves the
tracker trusting the cache, and every drop of models is followed by a memory release.
"""
import os

import pytest

from seedvr2_neo.loaded_models import LoadedModelTracker
from seedvr2_neo.model_lifecycle import DIT_CACHE_ID, VAE_CACHE_ID, ModelLifecycle

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"
BOTH = {DIT_CACHE_ID, VAE_CACHE_ID}


class CacheError(RuntimeError):
    pass


class FakeCache:
    """The vendored GlobalModelCache's remove_* contract; `fail_on` makes one remove raise after nothing was removed."""

    def __init__(self, cached=(), fail_on=None):
        self.cached = set(cached)
        self.removed = []
        self.fail_on = fail_on

    def remove_dit(self, dit_config):
        return self._remove("dit", dit_config["node_id"])

    def remove_vae(self, vae_config):
        return self._remove("vae", vae_config["node_id"])

    def _remove(self, kind, node_id):
        self.removed.append(node_id)
        if self.fail_on == kind:
            raise CacheError(f"remove_{kind} failed")
        if node_id in self.cached:
            self.cached.discard(node_id)
            return True
        return False


class Releases:
    def __init__(self, fail=False):
        self.count = 0
        self.fail = fail

    def __call__(self):
        self.count += 1
        if self.fail:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")


def make_lifecycle(release=None):
    release = release or Releases()
    return ModelLifecycle(LoadedModelTracker(), release), release


def make_models(folder, dit_bytes=b"dit", vae_bytes=b"vae"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / DIT).write_bytes(dit_bytes)
    (folder / VAE).write_bytes(vae_bytes)
    return str(folder / DIT), str(folder / VAE)


def kept_run(life, dit, vae):
    """A successful "Keep in RAM" run that leaves both models in the cache, as the pipeline does."""
    cache = FakeCache()
    life.prepare_run(cache, dit, vae, force_reload=False)
    cache.cached = set(BOTH)
    assert life.finish_run(cache, keep_models=True, succeeded=True) is True
    return cache


def bump_mtime(path, seconds=5):
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + seconds * 1_000_000_000))


# --- call order ----------------------------------------------------------------------------------------------------

def test_second_finish_after_one_prepare_keeps_nothing(tmp_path):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()
    cache = kept_run(life, dit, vae)

    # a stray second finish (e.g. a retried finally) must not re-remember anything
    assert life.finish_run(cache, keep_models=True, succeeded=True) is False
    assert cache.cached == set()
    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


def test_finish_without_any_prepare_never_trusts_later(tmp_path):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()

    assert life.finish_run(FakeCache(BOTH), keep_models=True, succeeded=True) is False
    assert release.count == 1
    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


def test_repeated_prepare_without_finish_remembers_the_last_files(tmp_path):
    first = make_models(tmp_path / "first", dit_bytes=b"first")
    second = make_models(tmp_path / "second", dit_bytes=b"second!")
    life, _ = make_lifecycle()
    cache = FakeCache()

    life.prepare_run(cache, *first, force_reload=False)
    life.prepare_run(cache, *second, force_reload=False)  # the first run died before its finish
    cache.cached = set(BOTH)
    life.finish_run(cache, keep_models=True, succeeded=True)

    assert life.prepare_run(FakeCache(BOTH), *first, force_reload=False) is True
    life.finish_run(FakeCache(), keep_models=False, succeeded=True)
    kept_run(life, *second)
    assert life.prepare_run(FakeCache(BOTH), *second, force_reload=False) is False


def test_repeated_prepare_with_trusted_files_keeps_the_cache_both_times(tmp_path):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()
    kept_run(life, dit, vae)
    cache = FakeCache(BOTH)
    releases_before = release.count

    assert life.prepare_run(cache, dit, vae, force_reload=False) is False
    assert life.prepare_run(cache, dit, vae, force_reload=False) is False
    assert cache.cached == BOTH
    assert release.count == releases_before
    assert life.finish_run(cache, keep_models=True, succeeded=True) is True


def test_force_reload_on_an_empty_cache_drops_nothing_and_releases_nothing(tmp_path):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()

    assert life.prepare_run(FakeCache(), dit, vae, force_reload=True) is False
    assert release.count == 0


@pytest.mark.parametrize("keep_models, succeeded", [(None, True), (True, None), (0, True), ("", True)])
def test_falsy_flags_mean_unload(tmp_path, keep_models, succeeded):
    dit, vae = make_models(tmp_path)
    life, _ = make_lifecycle()
    cache = FakeCache()
    life.prepare_run(cache, dit, vae, force_reload=False)
    cache.cached = set(BOTH)

    assert life.finish_run(cache, keep_models=keep_models, succeeded=succeeded) is False
    assert cache.cached == set()


# --- files changed between prepare and finish, or between runs ---------------------------------------------------

@pytest.mark.parametrize("change", ["rewrite_dit", "rewrite_vae", "touch_dit", "touch_vae", "delete_vae",
                                   "delete_dit_and_recreate"])
def test_files_changed_during_a_kept_run_are_not_trusted_next_time(tmp_path, change):
    dit, vae = make_models(tmp_path)
    life, _ = make_lifecycle()
    cache = FakeCache()
    life.prepare_run(cache, dit, vae, force_reload=False)

    if change == "rewrite_dit":
        with open(dit, "wb") as f:
            f.write(b"a newer, bigger dit")
    elif change == "rewrite_vae":
        with open(vae, "wb") as f:
            f.write(b"a newer, bigger vae")
    elif change == "touch_dit":
        bump_mtime(dit)
    elif change == "touch_vae":
        bump_mtime(vae)
    elif change == "delete_vae":
        os.remove(vae)
    else:
        os.remove(dit)
        with open(dit, "wb") as f:
            f.write(b"re-downloaded")

    cache.cached = set(BOTH)
    life.finish_run(cache, keep_models=True, succeeded=True)

    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


def test_file_missing_at_prepare_is_never_trusted_even_after_a_kept_run(tmp_path):
    dit, vae = make_models(tmp_path)
    os.remove(dit)
    life, _ = make_lifecycle()
    cache = FakeCache()
    life.prepare_run(cache, dit, vae, force_reload=False)
    cache.cached = set(BOTH)
    life.finish_run(cache, keep_models=True, succeeded=True)  # the pipeline found it some other way
    with open(dit, "wb") as f:
        f.write(b"dit")

    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


def test_same_files_under_another_spelling_are_reused(tmp_path):
    dit, vae = make_models(tmp_path / "models" / "SeedVR2")
    life, _ = make_lifecycle()
    kept_run(life, dit, vae)
    odd_dit = os.path.join(str(tmp_path), "models", "x", "..", "SeedVR2", DIT)

    assert life.prepare_run(FakeCache(BOTH), odd_dit, vae, force_reload=False) is False


def test_dit_and_vae_paths_swapped_drop_the_cache(tmp_path):
    dit, vae = make_models(tmp_path)
    life, _ = make_lifecycle()
    kept_run(life, dit, vae)

    assert life.prepare_run(FakeCache(BOTH), vae, dit, force_reload=False) is True


@pytest.mark.parametrize("folder_name", ["Модели", "models 🧠 ✨", "x" * 120])
def test_unicode_and_long_folders_keep_and_reuse(tmp_path, folder_name):
    try:
        dit, vae = make_models(tmp_path / folder_name / ("y" * 100))
    except OSError:
        pytest.skip("the OS cannot create this path here")
    life, _ = make_lifecycle()
    kept_run(life, dit, vae)

    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is False


@pytest.mark.parametrize("dit_path, vae_path", [("", ""), (".", "."), ("missing_dit", "missing_vae")])
def test_odd_paths_never_raise_and_never_trust(tmp_path, monkeypatch, dit_path, vae_path):
    monkeypatch.chdir(tmp_path)
    life, _ = make_lifecycle()
    cache = FakeCache()
    life.prepare_run(cache, dit_path, vae_path, force_reload=False)
    cache.cached = set(BOTH)
    life.finish_run(cache, keep_models=True, succeeded=True)

    # "" resolves to the working directory like ".": a folder has an identity and stays trusted while it is untouched
    if dit_path in ("", "."):
        (tmp_path / "new_file").write_bytes(b"x")
        bump_mtime(str(tmp_path))
    assert life.prepare_run(FakeCache(BOTH), dit_path, vae_path, force_reload=False) is True


# --- a cache whose remove_* raises ---------------------------------------------------------------------------------

@pytest.mark.parametrize("fail_on", ["dit", "vae"])
def test_failing_remove_in_prepare_leaves_nothing_trusted(tmp_path, fail_on):
    dit, vae = make_models(tmp_path)
    life, _ = make_lifecycle()
    kept_run(life, dit, vae)
    bump_mtime(dit)  # the next run must drop the cache

    with pytest.raises(CacheError):
        life.prepare_run(FakeCache(BOTH, fail_on=fail_on), dit, vae, force_reload=False)

    # the files changed back is impossible, but even the exact remembered files are no longer trusted
    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


@pytest.mark.parametrize("fail_on", ["dit", "vae"])
def test_failing_remove_in_finish_leaves_nothing_trusted(tmp_path, fail_on):
    dit, vae = make_models(tmp_path)
    life, _ = make_lifecycle()
    life.prepare_run(FakeCache(), dit, vae, force_reload=False)

    with pytest.raises(CacheError):
        life.finish_run(FakeCache(BOTH, fail_on=fail_on), keep_models=True, succeeded=False)

    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


def test_failing_dit_remove_still_removes_the_vae_and_releases(tmp_path):
    life, release = make_lifecycle()
    cache = FakeCache(BOTH, fail_on="dit")

    try:
        life.unload(cache)
    except CacheError:
        pass

    assert VAE_CACHE_ID not in cache.cached
    assert release.count == 1


# --- release_memory that raises ------------------------------------------------------------------------------------

def test_failing_release_in_prepare_leaves_the_cache_empty_and_untrusted(tmp_path):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()
    kept_run(life, dit, vae)
    release.fail = True
    cache = FakeCache(BOTH)

    with pytest.raises(RuntimeError):
        life.prepare_run(cache, dit, vae, force_reload=True)

    assert cache.cached == set()
    release.fail = False
    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


@pytest.mark.parametrize("keep_models, succeeded", [(False, True), (True, False)])
def test_failing_release_in_an_unloading_finish_leaves_the_cache_empty_and_untrusted(tmp_path, keep_models, succeeded):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()
    life.prepare_run(FakeCache(), dit, vae, force_reload=False)
    release.fail = True
    cache = FakeCache(BOTH)

    with pytest.raises(RuntimeError):
        life.finish_run(cache, keep_models=keep_models, succeeded=succeeded)

    assert cache.cached == set()
    release.fail = False
    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is True


def test_failing_release_in_a_keeping_finish_still_consumes_the_pending_run(tmp_path):
    dit, vae = make_models(tmp_path)
    life, release = make_lifecycle()
    life.prepare_run(FakeCache(), dit, vae, force_reload=False)
    release.fail = True

    with pytest.raises(RuntimeError):
        life.finish_run(FakeCache(BOTH), keep_models=True, succeeded=True)

    release.fail = False
    # the models stayed cached from a successful run with these exact files: reusing them is correct
    assert life.prepare_run(FakeCache(BOTH), dit, vae, force_reload=False) is False
    # and the pending run was consumed: a stray finish keeps nothing
    life.finish_run(FakeCache(), keep_models=True, succeeded=True)
    assert life.finish_run(FakeCache(BOTH), keep_models=True, succeeded=True) is False


# --- volume --------------------------------------------------------------------------------------------------------

def test_many_runs_release_once_per_finish_and_trust_only_the_last_kept_files(tmp_path):
    pairs = [make_models(tmp_path / f"m{i}", dit_bytes=b"d" * (i + 1)) for i in range(5)]
    life, release = make_lifecycle()
    cache = FakeCache()
    expected_releases = 0
    trusted = None  # the files the cached models came from, per the rules

    for step in range(500):
        pair = pairs[(step * 3) % len(pairs) if step % 4 else (step // 4) % 2]
        force = step % 13 == 0
        keep, succeeded = step % 3 != 0, step % 11 != 0

        had_models = bool(cache.cached)
        dropped = life.prepare_run(cache, *pair, force_reload=force)
        assert dropped is (had_models and (force or pair != trusted)), step
        expected_releases += dropped

        cache.cached = set(BOTH)  # the pipeline loaded or reused both
        kept = life.finish_run(cache, keep_models=keep, succeeded=succeeded)
        expected_releases += 1  # exactly one release per finish, kept or dropped
        assert kept is (keep and succeeded), step
        assert release.count == expected_releases, step
        assert cache.cached == (BOTH if kept else set()), step
        trusted = pair if kept else None

    if trusted is None:
        assert all(life.prepare_run(FakeCache(BOTH), *pair, force_reload=False) for pair in pairs)
    else:
        assert life.prepare_run(FakeCache(BOTH), *trusted, force_reload=False) is False

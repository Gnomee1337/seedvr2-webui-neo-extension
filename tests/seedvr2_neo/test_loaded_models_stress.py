"""Hostile paths and file changes for `seedvr2_neo/loaded_models.py`.

The tracker must err on the side of reloading: any doubt about a file means the cache is not trusted. It must also not
reload needlessly when the same file is reached through another spelling of its path.
"""
import os
import shutil
import sys

import pytest

from seedvr2_neo import loaded_models as lm
from seedvr2_neo.loaded_models import LoadedModelTracker, ModelFileIdentity, read_model_file_identity

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"


def make_models(folder, dit_bytes=b"dit", vae_bytes=b"vae"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / DIT).write_bytes(dit_bytes)
    (folder / VAE).write_bytes(vae_bytes)
    return str(folder / DIT), str(folder / VAE)


def trusted_after_remember(remembered, asked):
    tracker = LoadedModelTracker()
    tracker.remember(*[read_model_file_identity(path) for path in remembered])
    return tracker.is_cache_valid_for(*[read_model_file_identity(path) for path in asked])


# --- odd but valid paths -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("folder_name", ["Модели", "モデル", "models 🧠 ✨", "a b%20c", "dots.in.name", "trailing dot."])
def test_unicode_and_odd_folders_are_trusted_when_unchanged(tmp_path, folder_name):
    dit, vae = make_models(tmp_path / folder_name)

    assert trusted_after_remember((dit, vae), (dit, vae)) is True


def test_very_long_path_is_trusted_or_never_trusted_but_never_raises(tmp_path):
    folder = tmp_path
    for _ in range(12):
        folder = folder / ("deep_" + "x" * 20)
    try:
        dit, vae = make_models(folder)
    except OSError:
        pytest.skip("the OS cannot create paths this long here")
    assert len(dit) > 260

    identity = read_model_file_identity(dit)

    if identity is None:  # no long-path support: the cache is simply never trusted, which is safe
        assert trusted_after_remember((dit, vae), (dit, vae)) is False
    else:
        assert identity.size == 3
        assert trusted_after_remember((dit, vae), (dit, vae)) is True


def test_same_file_through_dotdot_and_doubled_separators_is_trusted(tmp_path):
    dit, vae = make_models(tmp_path / "models" / "SeedVR2")
    odd_dit = os.path.join(str(tmp_path), "models", "other", "..", "SeedVR2", ".", DIT)
    odd_vae = str(tmp_path) + os.sep + "models" + os.sep * 2 + "SeedVR2" + os.sep + VAE

    assert trusted_after_remember((dit, vae), (odd_dit, odd_vae)) is True


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive paths are a Windows property")
def test_path_case_does_not_force_a_reload_on_windows(tmp_path):
    dit, vae = make_models(tmp_path)

    assert trusted_after_remember((dit, vae), (dit.upper(), vae.swapcase())) is True


def test_symlink_to_the_same_file_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path / "real")
    link_folder = tmp_path / "link"
    link_folder.mkdir()
    link = link_folder / DIT
    try:
        os.symlink(dit, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges here")

    # Another path means "another file" for the tracker: a reload, never a wrong reuse.
    assert trusted_after_remember((dit, vae), (str(link), vae)) is False


def test_bytes_path_never_matches_a_str_path(tmp_path):
    dit, vae = make_models(tmp_path)

    assert trusted_after_remember((dit, vae), (os.fsencode(dit), vae)) is False


# --- what is not a readable model file -----------------------------------------------------------------------------

def test_directory_in_place_of_the_model_has_an_identity_but_changes_are_caught(tmp_path):
    folder = tmp_path / DIT
    folder.mkdir()
    _, vae = make_models(tmp_path / "vae")
    tracker = LoadedModelTracker()
    tracker.remember(read_model_file_identity(str(folder)), read_model_file_identity(vae))

    (folder / "new_entry").write_bytes(b"x")
    stat = os.stat(folder)
    os.utime(folder, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    assert tracker.is_cache_valid_for(read_model_file_identity(str(folder)), read_model_file_identity(vae)) is False


@pytest.mark.parametrize("error", [PermissionError(13, "Access is denied"), OSError(5, "I/O error"),
                                   FileNotFoundError(2, "gone between abspath and stat")])
def test_unreadable_file_has_no_identity(tmp_path, monkeypatch, error):
    dit, _ = make_models(tmp_path)

    def failing_stat(*args, **kwargs):
        raise error

    monkeypatch.setattr(lm.os, "stat", failing_stat)

    assert read_model_file_identity(dit) is None


@pytest.mark.parametrize("path", ["", ".", os.sep])
def test_empty_or_folder_like_paths_never_raise(path):
    read_model_file_identity(path)  # a folder identity or None; the loader then fails and the run unloads


def test_path_with_nul_has_no_identity(tmp_path):
    assert read_model_file_identity(str(tmp_path / "bad\0name.safetensors")) is None


# --- files changed between runs ------------------------------------------------------------------------------------

def test_file_replaced_by_a_copy_of_another_model_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path / "a", dit_bytes=b"model one")
    other_dit, _ = make_models(tmp_path / "b", dit_bytes=b"model two!")
    tracker = LoadedModelTracker()
    tracker.remember(read_model_file_identity(dit), read_model_file_identity(vae))

    shutil.copy2(other_dit, dit)  # keeps the other file's mtime

    assert tracker.is_cache_valid_for(read_model_file_identity(dit), read_model_file_identity(vae)) is False


def test_file_moved_over_by_os_replace_with_another_mtime_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(read_model_file_identity(dit), read_model_file_identity(vae))
    replacement = tmp_path / "download.tmp"
    replacement.write_bytes(b"DIT")  # same size, different content
    stat = os.stat(dit)
    # 10 ms later: NTFS keeps mtime in 100 ns steps (a +1 ns change is lost), exFAT in 10 ms steps
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))

    os.replace(replacement, dit)

    assert tracker.is_cache_valid_for(read_model_file_identity(dit), read_model_file_identity(vae)) is False


def test_vae_deleted_and_recreated_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(read_model_file_identity(dit), read_model_file_identity(vae))
    stat = os.stat(vae)
    os.remove(vae)
    with open(vae, "wb") as f:
        f.write(b"vae plus")

    assert tracker.is_cache_valid_for(read_model_file_identity(dit), read_model_file_identity(vae)) is False
    assert os.stat(vae).st_size != stat.st_size


# --- tracker state -------------------------------------------------------------------------------------------------

def test_swapped_dit_and_vae_are_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path)

    assert trusted_after_remember((dit, vae), (vae, dit)) is False


@pytest.mark.parametrize("remembered", [(None, None), ("dit", None), (None, "vae")])
def test_partial_memory_trusts_nothing_even_for_matching_nones(tmp_path, remembered):
    dit, vae = make_models(tmp_path)
    by_name = {"dit": read_model_file_identity(dit), "vae": read_model_file_identity(vae), None: None}
    tracker = LoadedModelTracker()
    tracker.remember(by_name[remembered[0]], by_name[remembered[1]])

    assert tracker.is_cache_valid_for(by_name[remembered[0]], by_name[remembered[1]]) is False
    assert tracker.is_cache_valid_for(by_name["dit"], by_name["vae"]) is False


def test_last_remember_wins_and_forget_twice_is_harmless(tmp_path):
    first = make_models(tmp_path / "first", dit_bytes=b"first dit")
    second = make_models(tmp_path / "second", dit_bytes=b"second dit!")
    tracker = LoadedModelTracker()
    tracker.remember(*map(read_model_file_identity, first))
    tracker.remember(*map(read_model_file_identity, second))

    assert tracker.is_cache_valid_for(*map(read_model_file_identity, first)) is False
    assert tracker.is_cache_valid_for(*map(read_model_file_identity, second)) is True
    tracker.forget()
    tracker.forget()
    assert tracker.is_cache_valid_for(*map(read_model_file_identity, second)) is False


def test_identity_compares_every_field():
    base = ModelFileIdentity(path="c:\\m\\dit.safetensors", size=10, mtime_ns=5)
    tracker = LoadedModelTracker()
    vae = ModelFileIdentity(path="c:\\m\\vae.safetensors", size=1, mtime_ns=1)
    tracker.remember(base, vae)

    for changed in (ModelFileIdentity(base.path + " ", 10, 5), ModelFileIdentity(base.path, 11, 5),
                    ModelFileIdentity(base.path, 10, 6), ModelFileIdentity(base.path, 10, 4)):
        assert tracker.is_cache_valid_for(changed, vae) is False
    assert tracker.is_cache_valid_for(ModelFileIdentity(base.path, 10, 5), vae) is True


def test_many_files_and_many_remembers_keep_only_the_last(tmp_path):
    folders = [make_models(tmp_path / f"m{i}", dit_bytes=b"d" * (i + 1)) for i in range(200)]
    tracker = LoadedModelTracker()
    for pair in folders:
        tracker.remember(*map(read_model_file_identity, pair))

    trusted = [tracker.is_cache_valid_for(*map(read_model_file_identity, pair)) for pair in folders]

    assert trusted == [False] * 199 + [True]

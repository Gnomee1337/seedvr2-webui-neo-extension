"""Hostile inputs for `seedvr2_neo/model_discovery.py`: missing folders, files in place of folders, odd names."""
import os
import sys

import pytest

from seedvr2_neo import model_discovery as md

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"


def _model_folder(parent, name="SeedVR2", files=()):
    folder = parent / name
    folder.mkdir(parents=True)
    for file_name in files:
        (folder / file_name).write_bytes(b"x")
    return folder


def _full_scan(tmp_path):
    _model_folder(tmp_path, files=[DIT, VAE])
    return md.scan_models(str(tmp_path), str(tmp_path / "ext"))


def test_missing_parents_give_no_folder(tmp_path):
    missing = tmp_path / "does-not-exist"

    scan = md.scan_models(str(missing), str(missing))

    assert scan.folder is None and scan.dit_models == () and scan.vae_models == ()


def test_file_named_like_the_folder_is_ignored(tmp_path):
    (tmp_path / "SeedVR2").write_bytes(b"not a folder")

    assert md.find_subfolders_ignoring_case(str(tmp_path), "SeedVR2") == []


def test_listing_a_missing_folder_is_empty(tmp_path):
    assert md.list_model_files(str(tmp_path / "nope")) == []


def test_names_with_spaces_and_unicode_are_listed(tmp_path):
    folder = tmp_path / "SeedVR2"
    folder.mkdir()
    for name in ["my model 7B.safetensors", "модель_vae.safetensors"]:
        (folder / name).write_bytes(b"x")

    scan = md.scan_models(str(tmp_path), str(tmp_path / "ext"))

    assert scan.dit_models == ("my model 7B.safetensors",)
    assert scan.vae_models == ("модель_vae.safetensors",)


def test_only_vae_files_leave_dit_empty(tmp_path):
    folder = tmp_path / "SeedVR2"
    folder.mkdir()
    (folder / "ema_vae_fp16.safetensors").write_bytes(b"x")

    scan = md.scan_models(str(tmp_path), str(tmp_path / "ext"))

    assert scan.dit_models == () and scan.default_dit is None
    assert "DiT" in md.describe_missing_models(scan)


# --- Odd entries inside the model folder -------------------------------------------------------------

def test_directories_named_like_model_files_are_not_listed(tmp_path):
    folder = _model_folder(tmp_path, files=[DIT])
    (folder / "dir_7b.safetensors").mkdir()
    (folder / "dir_vae.gguf").mkdir()

    assert md.list_model_files(str(folder)) == [DIT]


@pytest.mark.parametrize(
    "name",
    [
        "model.safetensors.part",  # unfinished download
        "model.safetensors.tmp",
        "model.SAFETENSORS",  # the loader only knows the lower-case suffixes
        "model.Gguf",
        "model.safetensors.bak",
        "model_safetensors",
        "model.ckpt",
        "model.pth",
    ],
)
def test_wrong_extensions_are_not_listed(tmp_path, name):
    folder = _model_folder(tmp_path, files=[name])

    assert md.list_model_files(str(folder)) == []


def test_zero_byte_model_files_are_listed_without_raising(tmp_path):
    folder = tmp_path / "SeedVR2"
    folder.mkdir()
    (folder / DIT).write_bytes(b"")
    (folder / VAE).write_bytes(b"")

    scan = md.scan_models(str(tmp_path), str(tmp_path / "ext"))

    assert scan.dit_models == (DIT,) and scan.vae_models == (VAE,)
    assert md.find_selection_error(scan, DIT, VAE) == ""


def test_folder_with_only_wrong_extensions_does_not_hide_the_fallback(tmp_path):
    models_path = tmp_path / "webui" / "models"
    _model_folder(models_path, files=["model.ckpt", "model.SAFETENSORS", "readme.txt"])
    extension_root = tmp_path / "ext"
    fallback = _model_folder(extension_root / "models", files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder == os.path.abspath(str(fallback))


def test_many_files_are_listed_sorted(tmp_path):
    folder = tmp_path / "SeedVR2"
    folder.mkdir()
    names = [f"m{index:04d}.safetensors" for index in range(1500)] + [f"x{index}.txt" for index in range(500)]
    for name in names:
        (folder / name).write_bytes(b"")

    listed = md.list_model_files(str(folder))

    assert len(listed) == 1500 and listed == sorted(listed, key=str.lower)


def test_file_vanishing_between_listing_and_check_is_skipped(tmp_path, monkeypatch):
    folder = _model_folder(tmp_path, files=[DIT])
    real_listdir = os.listdir
    monkeypatch.setattr(md.os, "listdir", lambda path: real_listdir(path) + ["ghost_7b.safetensors"])

    assert md.list_model_files(str(folder)) == [DIT]


def test_unreadable_folder_is_treated_as_empty_and_fallback_used(tmp_path, monkeypatch):
    models_path = tmp_path / "webui" / "models"
    locked = _model_folder(models_path, files=[DIT, VAE])
    extension_root = tmp_path / "ext"
    fallback = _model_folder(extension_root / "models", files=[DIT, VAE])
    real_listdir = os.listdir

    def listdir(path):
        if os.path.abspath(path) == os.path.abspath(str(locked)):
            raise PermissionError(13, "Access is denied", path)
        return real_listdir(path)

    monkeypatch.setattr(md.os, "listdir", listdir)

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder == os.path.abspath(str(fallback))


def test_unreadable_parent_gives_no_folder(tmp_path, monkeypatch):
    def listdir(path):
        raise PermissionError(13, "Access is denied", path)

    monkeypatch.setattr(md.os, "listdir", listdir)

    scan = md.scan_models(str(tmp_path / "webui" / "models"), str(tmp_path / "ext"))

    assert scan.folder is None
    assert scan.expected_folder in md.describe_missing_models(scan)


def test_symlinked_model_file_is_listed_and_broken_link_is_not(tmp_path):
    folder = _model_folder(tmp_path)
    target = tmp_path / "elsewhere" / DIT
    target.parent.mkdir()
    target.write_bytes(b"x")
    try:
        os.symlink(str(target), str(folder / DIT))
        os.symlink(str(tmp_path / "missing.safetensors"), str(folder / "broken_7b.safetensors"))
    except (OSError, NotImplementedError) as exc:  # Windows without Developer Mode or admin rights
        pytest.skip(f"symlinks not permitted here: {exc}")

    assert md.list_model_files(str(folder)) == [DIT]


def test_model_folder_reached_through_a_link_is_used(tmp_path):
    real = _model_folder(tmp_path / "other_drive", name="seedvr2_models", files=[DIT, VAE])
    models_path = tmp_path / "webui" / "models"
    models_path.mkdir(parents=True)
    link = models_path / "SeedVR2"
    try:
        if sys.platform == "win32":
            import _winapi  # a directory junction needs no admin rights, unlike a symlink

            _winapi.CreateJunction(str(real), str(link))
        else:
            os.symlink(str(real), str(link), target_is_directory=True)
    except (OSError, AttributeError, ImportError) as exc:
        pytest.skip(f"directory links not available here: {exc}")

    scan = md.scan_models(str(models_path), str(tmp_path / "ext"))

    assert scan.folder == os.path.abspath(str(link))
    assert scan.dit_models == (DIT,) and scan.vae_models == (VAE,)


# --- Odd folder names and parents ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "Seed VR2",
        "SeedVR2.bak",
        "SeedVR2_old",
        "SeedVR",
        "SееdVR2",  # Cyrillic look-alike letters
        "ＳｅｅｄＶＲ２",  # full-width letters
    ],
)
def test_near_miss_folder_names_are_not_used(tmp_path, name):
    models_path = tmp_path / "webui" / "models"
    _model_folder(models_path, name=name, files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(tmp_path / "ext"))

    assert scan.folder is None
    assert scan.expected_folder == os.path.abspath(str(models_path / "SeedVR2"))


@pytest.mark.skipif(sys.platform == "win32", reason="Windows strips trailing spaces from folder names")
def test_trailing_space_folder_name_is_not_used(tmp_path):
    models_path = tmp_path / "webui" / "models"
    _model_folder(models_path, name="SeedVR2 ", files=[DIT, VAE])

    assert md.scan_models(str(models_path), str(tmp_path / "ext")).folder is None


@pytest.mark.parametrize("name", ["seedvr2", "SEEDVR2", "sEeDvR2"])
def test_case_variants_are_found(tmp_path, name):
    models_path = tmp_path / "webui" / "models"
    _model_folder(models_path, name=name, files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(tmp_path / "ext"))

    assert scan.folder is not None and os.path.basename(scan.folder).lower() == "seedvr2"
    assert scan.dit_models == (DIT,)


def test_models_path_that_is_a_file_does_not_raise(tmp_path):
    models_file = tmp_path / "models"
    models_file.write_bytes(b"not a folder")
    extension_root = tmp_path / "ext"
    fallback = _model_folder(extension_root / "models", files=[DIT, VAE])

    scan = md.scan_models(str(models_file), str(extension_root))

    assert scan.folder == os.path.abspath(str(fallback))


def test_seedvr2_file_in_forge_folder_does_not_hide_extension_folder(tmp_path):
    models_path = tmp_path / "webui" / "models"
    models_path.mkdir(parents=True)
    (models_path / "SeedVR2").write_bytes(b"a file, not a folder")
    extension_root = tmp_path / "ext"
    fallback = _model_folder(extension_root / "models", files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder == os.path.abspath(str(fallback))


def test_unicode_and_spaces_in_parent_path(tmp_path):
    models_path = tmp_path / "Пользователь 🎨" / "web ui" / "models"
    _model_folder(models_path, files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(tmp_path / "ext"))

    assert scan.folder == os.path.abspath(str(models_path / "SeedVR2"))
    assert scan.folder in md.find_selection_error(scan, "missing.safetensors", VAE)


# --- Selections that cannot be used -------------------------------------------------------------------

@pytest.mark.parametrize("dit", [None, ""])
def test_empty_dit_selection_is_an_error(tmp_path, dit):
    assert md.find_selection_error(_full_scan(tmp_path), dit, VAE) == "No DiT model selected."


@pytest.mark.parametrize("vae", [None, ""])
def test_empty_vae_selection_is_an_error(tmp_path, vae):
    assert md.find_selection_error(_full_scan(tmp_path), DIT, vae) == "No VAE model selected."


@pytest.mark.parametrize(
    "dit, vae",
    [
        ("   ", VAE),  # blank
        ("../" + DIT, VAE),  # path outside the folder
        (os.path.join("..", "..", "evil.safetensors"), VAE),
        (DIT.upper(), VAE),  # the same file on Windows, but not a listed name
        (VAE, VAE),  # VAE chosen as DiT
        (DIT, DIT),  # DiT chosen as VAE
        (VAE, DIT),  # swapped
        (DIT + " ", VAE),
        ("x" * 5000 + ".safetensors", VAE),  # very long name
        ("🎨.safetensors", VAE),
    ],
)
def test_selection_not_in_the_listing_is_an_error(tmp_path, dit, vae):
    error = md.find_selection_error(_full_scan(tmp_path), dit, vae)

    assert "not in" in error


def test_file_deleted_after_the_scan_is_caught_by_a_fresh_scan(tmp_path):
    folder = _model_folder(tmp_path, files=[DIT, "seedvr2_ema_7b_fp16.safetensors", VAE])
    stale = md.scan_models(str(tmp_path), str(tmp_path / "ext"))
    (folder / DIT).unlink()

    fresh = md.scan_models(str(tmp_path), str(tmp_path / "ext"))

    assert md.find_selection_error(stale, DIT, VAE) == ""  # a stale scan cannot know; run() rescans
    assert "not in" in md.find_selection_error(fresh, DIT, VAE)
    assert fresh.keep_or_default_dit(DIT) == "seedvr2_ema_7b_fp16.safetensors"


def test_whole_folder_deleted_after_the_scan(tmp_path):
    folder = _model_folder(tmp_path, files=[DIT, VAE])
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()

    fresh = md.scan_models(str(tmp_path), str(tmp_path / "ext"))

    assert fresh.folder is None
    assert md.find_selection_error(fresh, DIT, VAE).startswith("No SeedVR2 models found")


@pytest.mark.parametrize("current", [None, "", "gone.safetensors", ["list", "value"], 0])
def test_keep_or_default_never_raises_on_odd_current_values(tmp_path, current):
    scan = _full_scan(tmp_path)

    assert scan.keep_or_default_dit(current) == DIT
    assert scan.keep_or_default_vae(current) == VAE


def test_keep_or_default_on_an_empty_scan_gives_none(tmp_path):
    scan = md.scan_models(str(tmp_path / "nope"), str(tmp_path / "nope"))

    assert scan.keep_or_default_dit(DIT) is None and scan.keep_or_default_vae(VAE) is None
    assert scan.default_dit is None and scan.default_vae is None

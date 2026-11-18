"""Hostile inputs for the `find_model_file` local patch (`seedvr2_vendor/utils/constants.py`): odd names and folders."""
import os

import pytest

from seedvr2_vendor.utils.constants import find_model_file

NAME = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


@pytest.fixture
def empty_cwd(tmp_path, monkeypatch):
    """A working directory without `models/SEEDVR2`, so only the caller's folder can match."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def test_file_deleted_from_the_folder_gives_its_path_not_an_exception(tmp_path, empty_cwd):
    folder = tmp_path / "SeedVR2"
    folder.mkdir()

    assert find_model_file(NAME, str(folder)) == os.path.join(str(folder), NAME)


def test_directory_named_like_the_model_is_not_returned_as_found(tmp_path, monkeypatch):
    folder = tmp_path / "SeedVR2"
    (folder / NAME).mkdir(parents=True)
    cwd_copy = tmp_path / "cwd" / "models" / "SEEDVR2"
    cwd_copy.mkdir(parents=True)
    (cwd_copy / NAME).write_bytes(b"x")
    monkeypatch.chdir(tmp_path / "cwd")

    found = find_model_file(NAME, str(folder))

    assert os.path.isfile(found)


@pytest.mark.parametrize("fallback_dir", [None, ""])
def test_no_folder_given_uses_the_old_lookup(tmp_path, empty_cwd, fallback_dir):
    found = find_model_file(NAME, fallback_dir)

    assert found.endswith(NAME) and not os.path.isfile(found)


def test_unicode_and_spaces_in_the_folder(tmp_path, empty_cwd):
    folder = tmp_path / "Пользователь 🎨" / "Seed VR2"
    folder.mkdir(parents=True)
    (folder / NAME).write_bytes(b"x")

    assert find_model_file(NAME, str(folder)) == os.path.join(str(folder), NAME)


@pytest.mark.xfail(
    strict=True,
    raises=NotADirectoryError,
    reason="bug (vendored scan): when the file is missing from the chosen folder and `./models/SEEDVR2` "
           "in the working directory is a file, get_all_model_files lists it and raises",
)
def test_cwd_models_entry_that_is_a_file_does_not_raise(tmp_path, empty_cwd):
    (empty_cwd / "models").mkdir()
    (empty_cwd / "models" / "SEEDVR2").write_bytes(b"not a folder")
    folder = tmp_path / "SeedVR2"
    folder.mkdir()

    assert find_model_file(NAME, str(folder)) == os.path.join(str(folder), NAME)

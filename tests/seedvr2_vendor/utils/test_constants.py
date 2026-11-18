"""`find_model_file` local patch (`seedvr2_vendor/utils/constants.py`): the caller's folder wins over the cwd scan."""
from seedvr2_vendor.utils.constants import find_model_file

NAME = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def test_explicit_folder_wins_over_cwd_relative_folder(tmp_path, monkeypatch):
    cwd_folder = tmp_path / "cwd" / "models" / "SEEDVR2"
    cwd_folder.mkdir(parents=True)
    (cwd_folder / NAME).write_bytes(b"stale copy")
    chosen = tmp_path / "forge" / "models" / "SeedVR2"
    chosen.mkdir(parents=True)
    (chosen / NAME).write_bytes(b"chosen")
    monkeypatch.chdir(tmp_path / "cwd")

    assert find_model_file(NAME, str(chosen)) == str(chosen / NAME)


def test_without_file_in_folder_the_old_lookup_applies(tmp_path, monkeypatch):
    cwd_folder = tmp_path / "models" / "SEEDVR2"
    cwd_folder.mkdir(parents=True)
    (cwd_folder / NAME).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    found = find_model_file(NAME, str(tmp_path / "empty"))

    assert (tmp_path / "models" / "SEEDVR2" / NAME).samefile(found)

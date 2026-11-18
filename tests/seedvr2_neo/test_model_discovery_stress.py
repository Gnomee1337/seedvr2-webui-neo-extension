"""Hostile inputs for `seedvr2_neo/model_discovery.py`: missing folders, files in place of folders, odd names."""
from seedvr2_neo import model_discovery as md


def test_missing_parents_give_no_folder(tmp_path):
    missing = tmp_path / "does-not-exist"

    scan = md.scan_models(str(missing), str(missing))

    assert scan.folder is None and scan.dit_models == () and scan.vae_models == ()


def test_file_named_like_the_folder_is_ignored(tmp_path):
    (tmp_path / "SeedVR2").write_bytes(b"not a folder")

    assert md.find_subfolder_ignoring_case(str(tmp_path), "SeedVR2") is None


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

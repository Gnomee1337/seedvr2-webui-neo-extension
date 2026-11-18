"""File identity of cached models, `seedvr2_neo/loaded_models.py`."""
import os

from seedvr2_neo.loaded_models import LoadedModelTracker, read_model_file_identity

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"


def make_models(folder, dit_bytes=b"dit", vae_bytes=b"vae"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / DIT).write_bytes(dit_bytes)
    (folder / VAE).write_bytes(vae_bytes)
    return str(folder / DIT), str(folder / VAE)


def test_empty_tracker_trusts_nothing(tmp_path):
    dit, vae = make_models(tmp_path)

    assert LoadedModelTracker().is_cache_valid_for(dit, vae) is False


def test_same_files_are_trusted(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(dit, vae)

    assert tracker.is_cache_valid_for(dit, vae) is True


def test_same_name_in_another_folder_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path / "webui" / "models" / "SeedVR2")
    other_dit, other_vae = make_models(tmp_path / "ext" / "models")
    tracker = LoadedModelTracker()
    tracker.remember(dit, vae)

    assert tracker.is_cache_valid_for(other_dit, vae) is False
    assert tracker.is_cache_valid_for(dit, other_vae) is False


def test_file_replaced_in_place_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(dit, vae)

    with open(dit, "wb") as f:
        f.write(b"another model of another size")

    assert tracker.is_cache_valid_for(dit, vae) is False


def test_file_with_same_size_but_new_mtime_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(dit, vae)

    stat = os.stat(dit)
    os.utime(dit, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert tracker.is_cache_valid_for(dit, vae) is False


def test_deleted_file_is_not_trusted(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(dit, vae)

    os.remove(vae)

    assert tracker.is_cache_valid_for(dit, vae) is False


def test_forget_drops_trust(tmp_path):
    dit, vae = make_models(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(dit, vae)
    tracker.forget()

    assert tracker.is_cache_valid_for(dit, vae) is False


def test_remembering_missing_files_trusts_nothing(tmp_path):
    tracker = LoadedModelTracker()
    tracker.remember(str(tmp_path / "missing_dit"), str(tmp_path / "missing_vae"))

    assert tracker.is_cache_valid_for(str(tmp_path / "missing_dit"), str(tmp_path / "missing_vae")) is False


def test_relative_and_absolute_paths_match(tmp_path, monkeypatch):
    dit, vae = make_models(tmp_path)
    monkeypatch.chdir(tmp_path)
    tracker = LoadedModelTracker()
    tracker.remember(DIT, VAE)

    assert tracker.is_cache_valid_for(dit, vae) is True


def test_missing_path_has_no_identity(tmp_path):
    assert read_model_file_identity(str(tmp_path / "nope")) is None

"""Model folder resolution, listing and UI texts of `seedvr2_neo/model_discovery.py`."""
import os

from seedvr2_neo import model_discovery as md

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"


def make_folder(parent, name="SeedVR2", files=()):
    folder = parent / name
    folder.mkdir(parents=True)
    for file_name in files:
        (folder / file_name).write_bytes(b"x")
    return folder


def layout(tmp_path):
    """A Forge models folder and an extension root, both empty."""
    models_path = tmp_path / "webui" / "models"
    extension_root = tmp_path / "ext"
    models_path.mkdir(parents=True)
    (extension_root / "models").mkdir(parents=True)
    return models_path, extension_root


def test_forge_models_folder_is_used(tmp_path):
    models_path, extension_root = layout(tmp_path)
    folder = make_folder(models_path, files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder == os.path.abspath(folder)
    assert scan.dit_models == (DIT,)
    assert scan.vae_models == (VAE,)


def test_extension_folder_is_the_fallback(tmp_path):
    models_path, extension_root = layout(tmp_path)
    folder = make_folder(extension_root / "models", files=[DIT, VAE])

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder == os.path.abspath(folder)


def test_forge_folder_wins_when_both_have_models(tmp_path):
    models_path, extension_root = layout(tmp_path)
    forge_folder = make_folder(models_path, files=[DIT])
    make_folder(extension_root / "models", files=[VAE])

    assert md.find_model_folder(str(models_path), str(extension_root)) == os.path.abspath(forge_folder)


def test_empty_forge_folder_does_not_hide_extension_models(tmp_path):
    models_path, extension_root = layout(tmp_path)
    make_folder(models_path)
    ext_folder = make_folder(extension_root / "models", files=[DIT, VAE])

    assert md.find_model_folder(str(models_path), str(extension_root)) == os.path.abspath(ext_folder)


def test_folder_name_matched_ignoring_case_on_case_sensitive_fs(tmp_path, monkeypatch):
    models_path, extension_root = layout(tmp_path)
    real_folder = make_folder(models_path, name="seedvr2", files=[DIT, VAE])
    real_isdir = os.path.isdir

    def case_sensitive_isdir(path):
        parent, name = os.path.split(os.path.normpath(path))
        return real_isdir(path) and name in os.listdir(parent)

    monkeypatch.setattr(md.os.path, "isdir", case_sensitive_isdir)

    assert md.find_model_folder(str(models_path), str(extension_root)) == os.path.abspath(real_folder)


def fake_case_sensitive_fs(monkeypatch, tree):
    """Replace the file system seen by model_discovery with `tree`: {folder: [entries]}; files are leaves.

    Windows and macOS cannot hold `SeedVR2` and `seedvr2` side by side, so these tests fake a Linux one.
    """
    folders = {os.path.normpath(path): entries for path, entries in tree.items()}

    def listdir(path):
        try:
            return list(folders[os.path.normpath(path)])
        except KeyError:
            raise FileNotFoundError(path)

    monkeypatch.setattr(md.os, "listdir", listdir)
    monkeypatch.setattr(md.os.path, "isdir", lambda path: os.path.normpath(path) in folders)
    monkeypatch.setattr(md.os.path, "isfile", lambda path: os.path.normpath(path) not in folders)


def test_exact_spelling_comes_first_then_alphabetical(monkeypatch):
    parent = os.path.join("root", "models")
    fake_case_sensitive_fs(monkeypatch, {
        parent: ["seedvr2", "SEEDVR2", "SeedVR2", "other"],
        os.path.join(parent, "seedvr2"): [],
        os.path.join(parent, "SEEDVR2"): [],
        os.path.join(parent, "SeedVR2"): [],
    })

    found = md.find_subfolders_ignoring_case(parent, "SeedVR2")

    assert [os.path.basename(path) for path in found] == ["SeedVR2", "SEEDVR2", "seedvr2"]


def test_empty_exact_spelling_does_not_hide_other_spelling_with_models(monkeypatch):
    models_path = os.path.join("root", "models")
    extension_root = os.path.join("root", "ext")
    fake_case_sensitive_fs(monkeypatch, {
        models_path: ["SeedVR2", "seedvr2"],
        os.path.join(models_path, "SeedVR2"): [],
        os.path.join(models_path, "seedvr2"): [DIT, VAE],
    })

    found = md.find_model_folder(models_path, extension_root)

    assert found == os.path.abspath(os.path.join(models_path, "seedvr2"))


def test_path_keeps_the_real_spelling_of_the_folder(tmp_path):
    models_path, extension_root = layout(tmp_path)
    make_folder(models_path, name="seedvr2", files=[DIT, VAE])

    found = md.find_model_folder(str(models_path), str(extension_root))

    assert os.path.basename(found) == "seedvr2"


def test_complete_extension_folder_wins_over_partial_forge_folder(tmp_path):
    models_path, extension_root = layout(tmp_path)
    make_folder(models_path, files=[DIT])
    ext_folder = make_folder(extension_root / "models", files=[DIT, VAE])

    assert md.find_model_folder(str(models_path), str(extension_root)) == os.path.abspath(ext_folder)


def test_partial_forge_folder_is_used_when_no_folder_is_complete(tmp_path):
    models_path, extension_root = layout(tmp_path)
    forge_folder = make_folder(models_path, files=[DIT])
    make_folder(extension_root / "models", files=[VAE])

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder == os.path.abspath(forge_folder)
    assert "No VAE model" in md.describe_missing_models(scan)


def test_files_sorted_and_filtered(tmp_path):
    folder = make_folder(tmp_path, files=["b.gguf", "A.safetensors", "notes.txt", "model.ckpt", "x.GGUF"])
    (folder / "sub.safetensors").mkdir()

    assert md.list_model_files(str(folder)) == ["A.safetensors", "b.gguf"]


def test_preferred_vae_is_default(tmp_path):
    models_path, extension_root = layout(tmp_path)
    make_folder(models_path, files=[DIT, "another_vae.safetensors", VAE])

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.default_vae == VAE
    assert scan.default_dit == DIT


def test_refresh_keeps_existing_choice_and_replaces_missing():
    scan = md.ModelScan("f", "f", dit_models=("a", "b"), vae_models=(VAE,))

    assert scan.keep_or_default_dit("b") == "b"
    assert scan.keep_or_default_dit("gone") == "a"
    assert scan.keep_or_default_vae(None) == VAE


def test_note_names_expected_folder_when_nothing_found(tmp_path):
    models_path, extension_root = layout(tmp_path)

    scan = md.scan_models(str(models_path), str(extension_root))

    assert scan.folder is None
    assert md.expected_model_folder(str(models_path)) in md.describe_missing_models(scan)


def test_note_names_missing_kind():
    scan = md.ModelScan("/m/SeedVR2", "/m/SeedVR2", dit_models=(DIT,))

    note = md.describe_missing_models(scan)

    assert "VAE" in note and "DiT" not in note and "/m/SeedVR2" in note


def test_no_note_when_both_kinds_present():
    scan = md.ModelScan("/m", "/m", dit_models=(DIT,), vae_models=(VAE,))

    assert md.describe_missing_models(scan) == ""


def test_selection_error_cases():
    scan = md.ModelScan("/m", "/m", dit_models=(DIT,), vae_models=(VAE,))

    assert md.find_selection_error(scan, DIT, VAE) == ""
    assert "No DiT" in md.find_selection_error(scan, None, VAE)
    assert "No VAE" in md.find_selection_error(scan, DIT, "")
    assert "'old.gguf'" in md.find_selection_error(scan, "old.gguf", VAE)
    assert "No SeedVR2 models" in md.find_selection_error(md.ModelScan(None, "/m"), DIT, VAE)


def test_selection_error_names_the_missing_kind_when_folder_lacks_it():
    scan = md.ModelScan("/m", "/m", dit_models=(), vae_models=(VAE,))

    error = md.find_selection_error(scan, None, VAE)

    assert error.startswith("No DiT model in /m")

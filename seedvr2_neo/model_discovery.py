"""Find the SeedVR2 model folder and list the model files in it.

Pure file-system logic with no Forge or Gradio imports, so it is testable on its own.
`scripts/seedvr2_script.py` turns a `ModelScan` into dropdowns and a note, and passes
`ModelScan.folder` to the vendored pipeline (`src/`), which loads the chosen files from it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

MODEL_FOLDER_NAME = "SeedVR2"
# Case-sensitive on purpose: the vendored loader tells GGUF from safetensors by these exact suffixes.
MODEL_FILE_EXTENSIONS = (".safetensors", ".gguf")
VAE_NAME_MARKER = "vae"
PREFERRED_VAE_MARKER = "ema_vae_fp16"


@dataclass(frozen=True)
class ModelScan:
    """Result of one scan: the folder in use (None when no folder has models) and its models.

    `expected_folder` is where the user is told to put models when nothing is found.
    """

    folder: Optional[str]
    expected_folder: str
    dit_models: tuple[str, ...] = ()
    vae_models: tuple[str, ...] = ()

    @property
    def default_dit(self) -> Optional[str]:
        return self.dit_models[0] if self.dit_models else None

    @property
    def default_vae(self) -> Optional[str]:
        preferred = next((name for name in self.vae_models if PREFERRED_VAE_MARKER in name), None)
        return preferred or (self.vae_models[0] if self.vae_models else None)

    def keep_or_default_dit(self, current: Optional[str]) -> Optional[str]:
        """Keep the user's DiT choice after a rescan if the file is still there."""
        return current if current in self.dit_models else self.default_dit

    def keep_or_default_vae(self, current: Optional[str]) -> Optional[str]:
        """Keep the user's VAE choice after a rescan if the file is still there."""
        return current if current in self.vae_models else self.default_vae


def scan_models(models_path: str, extension_root: str) -> ModelScan:
    """Find the model folder and split its files into DiT and VAE models."""
    folder = find_model_folder(models_path, extension_root)
    files = list_model_files(folder) if folder else []
    return ModelScan(
        folder=folder,
        expected_folder=expected_model_folder(models_path),
        dit_models=tuple(name for name in files if not is_vae_model(name)),
        vae_models=tuple(name for name in files if is_vae_model(name)),
    )


def find_model_folder(models_path: str, extension_root: str) -> Optional[str]:
    """Return the absolute model folder, or None when no candidate holds a model file.

    Forge's models folder comes first (it is where users keep their other models);
    the extension's own `models/` is the fallback. A folder without model files is skipped,
    so an empty `SeedVR2` folder in one place does not hide models in the other.
    """
    for parent in _candidate_parents(models_path, extension_root):
        folder = find_subfolder_ignoring_case(parent, MODEL_FOLDER_NAME)
        if folder and list_model_files(folder):
            return os.path.abspath(folder)
    return None


def expected_model_folder(models_path: str) -> str:
    """The folder named to the user when no models are found."""
    return os.path.abspath(os.path.join(models_path, MODEL_FOLDER_NAME))


def find_subfolder_ignoring_case(parent: str, name: str) -> Optional[str]:
    """Find `parent/name` with any letter case, so Linux finds `seedvr2` like Windows does.

    The exact spelling wins; among other spellings the alphabetically first one is used.
    """
    exact = os.path.join(parent, name)
    if os.path.isdir(exact):
        return exact
    try:
        entries = sorted(os.listdir(parent))
    except OSError:
        return None
    for entry in entries:
        path = os.path.join(parent, entry)
        if entry.lower() == name.lower() and os.path.isdir(path):
            return path
    return None


def list_model_files(folder: str) -> list[str]:
    """Model file names in `folder`, sorted case-insensitively; [] if it cannot be read."""
    try:
        entries = os.listdir(folder)
    except OSError:
        return []
    models = (
        entry for entry in entries
        if entry.endswith(MODEL_FILE_EXTENSIONS) and os.path.isfile(os.path.join(folder, entry))
    )
    return sorted(models, key=str.lower)


def is_vae_model(file_name: str) -> bool:
    return VAE_NAME_MARKER in file_name.lower()


def describe_missing_models(scan: ModelScan) -> str:
    """User-facing note (Markdown); empty when both a DiT and a VAE model are available."""
    if scan.folder is None:
        return (
            f"⚠️ No SeedVR2 models found. Put the DiT and VAE files into `{scan.expected_folder}` "
            f"and press 🔄."
        )
    missing = _missing_kinds(scan)
    if not missing:
        return ""
    return f"⚠️ No {' and no '.join(missing)} model in `{scan.folder}`. Add the file and press 🔄."


def find_selection_error(scan: ModelScan, dit_model: Optional[str], vae_model: Optional[str]) -> str:
    """Why the selected models cannot be used, or "" when they can."""
    if scan.folder is None:
        return f"No SeedVR2 models found in {scan.expected_folder}."
    for kind, selected, available in (
        ("DiT", dit_model, scan.dit_models),
        ("VAE", vae_model, scan.vae_models),
    ):
        if not selected:
            return f"No {kind} model selected."
        if selected not in available:
            return f"{kind} model '{selected}' is not in {scan.folder}. Press 🔄 and choose again."
    return ""


def _candidate_parents(models_path: str, extension_root: str) -> Iterable[str]:
    return (models_path, os.path.join(extension_root, "models"))


def _missing_kinds(scan: ModelScan) -> list[str]:
    return [kind for kind, models in (("DiT", scan.dit_models), ("VAE", scan.vae_models)) if not models]

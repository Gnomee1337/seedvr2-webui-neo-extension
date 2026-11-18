"""The vendored package is named `seedvr2_vendor`, not upstream's `src`: no import or path string may still use the old name."""
import importlib.util
import os
import re
from pathlib import Path

import pytest

from seedvr2_vendor.utils.constants import get_vae_config_path

PACKAGE_DIR = Path(importlib.util.find_spec("seedvr2_vendor").origin).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# `from src...` / `from .src...`, `src` anywhere in an `import` list, or a string that is `src` or starts with
# `src/`, `src\` or `src.` (paths, dotted module names for importlib), or an unquoted YAML value that does.
OLD_NAME_REFERENCE = re.compile(
    r"""^\s*from\s+\.?src\b|^\s*import\s+[\w., ]*\bsrc\b|['"]src(['"/\\.])|:\s*src[/\\.]""", re.MULTILINE
)


def _checked_files():
    yield REPO_ROOT / "__init__.py"
    yield from (REPO_ROOT / "scripts").rglob("*.py")
    yield from (REPO_ROOT / "seedvr2_neo").rglob("*.py")
    yield from PACKAGE_DIR.rglob("*.py")
    yield from PACKAGE_DIR.rglob("*.yaml")
    yield from REPO_ROOT.glob("configs_*/*.yaml")


@pytest.mark.parametrize("path", list(_checked_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_reference_to_the_old_package_name(path):
    assert not OLD_NAME_REFERENCE.search(path.read_text(encoding="utf-8"))


def test_vae_config_path_points_to_the_packaged_yaml():
    path = Path(get_vae_config_path()).resolve()
    assert path.is_file()
    assert path.parent == PACKAGE_DIR / "models" / "video_vae_v3"


def test_vae_config_path_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    expected = get_vae_config_path()
    monkeypatch.chdir(tmp_path)
    assert get_vae_config_path() == expected
    assert os.path.isabs(expected)


@pytest.mark.parametrize("line", [
    "from src.core import x",
    "import src.utils.debug",
    "import os, src.utils",
    "from .src.interfaces import y",
    "path = 'src/models/a.yaml'",
    'path = "src\\\\models"',
    "os.path.join(root, 'src', 'models')",
    "importlib.import_module('src.utils')",
    'Path(root) / "src"',
    "  path: src.models.dit_3b.nadit",
    "  path: src/models/x.yaml",
    "__inherit__: src/models/a.yaml",
])
def test_pattern_catches_old_name_forms(line):
    assert OLD_NAME_REFERENCE.search(line)


@pytest.mark.parametrize("line", [
    "_broadcast_data(data, src, group)",
    "dist.broadcast(data, src=src)",
    "from .source import x",
    "mean_src = box_filter(src, radius)",
    "'srcset'",
])
def test_pattern_ignores_unrelated_src(line):
    assert not OLD_NAME_REFERENCE.search(line)

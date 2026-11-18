"""The vendored package is named `seedvr2_vendor`, not upstream's `src`: nothing may still reach it by the old name."""
import importlib.util
import re
from pathlib import Path

import pytest

PACKAGE_DIR = Path(importlib.util.find_spec("seedvr2_vendor").origin).parent
REPO_ROOT = PACKAGE_DIR.parent

# An import of the old package, or a path string that starts with the old folder name
OLD_NAME_REFERENCE = re.compile(r"""^\s*(from|import)\s+\.?src\b|['"]src[/\\]""", re.MULTILINE)


def _python_files():
    yield REPO_ROOT / "__init__.py"
    yield from (REPO_ROOT / "scripts").rglob("*.py")
    yield from (REPO_ROOT / "seedvr2_neo").rglob("*.py")
    yield from PACKAGE_DIR.rglob("*.py")


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_reference_to_the_old_package_name(path):
    assert not OLD_NAME_REFERENCE.search(path.read_text(encoding="utf-8"))


def test_vae_config_is_where_model_configuration_looks():
    # model_configuration.py builds this path from its own package folder
    assert (PACKAGE_DIR / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml").is_file()


def test_pattern_catches_the_old_forms():
    for line in ("from src.core import x", "import src.utils.debug", "from .src.interfaces import y",
                 "path = 'src/models/a.yaml'", 'path = "src\\\\models"'):
        assert OLD_NAME_REFERENCE.search(line), line
    for line in ("_broadcast_data(data, src, group)", "dist.broadcast(data, src=src)", "from .source import x"):
        assert not OLD_NAME_REFERENCE.search(line), line

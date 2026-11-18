"""Detection of another extension's `src` package (`seedvr2_neo/package_check.py`)."""
import os
from types import SimpleNamespace

from seedvr2_neo.package_check import describe_shadowed_package


def regular_package(folder):
    return SimpleNamespace(__file__=os.path.join(str(folder), "__init__.py"))


def make_src(root):
    folder = root / "src"
    folder.mkdir(parents=True)
    return folder


def test_our_package_gives_no_error(tmp_path):
    ours = make_src(tmp_path)

    assert describe_shadowed_package(regular_package(ours), str(tmp_path)) == ""


def test_relative_spelling_of_our_package_matches(tmp_path, monkeypatch):
    make_src(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert describe_shadowed_package(regular_package("src"), str(tmp_path)) == ""


def test_other_extensions_package_is_named_with_its_real_spelling(tmp_path):
    make_src(tmp_path / "seedvr2")
    foreign = make_src(tmp_path / "Other-Extension")

    error = describe_shadowed_package(regular_package(foreign), str(tmp_path / "seedvr2"))

    assert "Other-Extension" in error and "Disable that extension" in error


def test_namespace_package_is_reported_by_its_folder(tmp_path):
    make_src(tmp_path / "seedvr2")
    foreign = make_src(tmp_path / "other")

    error = describe_shadowed_package(SimpleNamespace(__path__=[str(foreign)]), str(tmp_path / "seedvr2"))

    assert str(foreign) in error


def test_package_without_location_is_reported(tmp_path):
    error = describe_shadowed_package(SimpleNamespace(), str(tmp_path))

    assert "unknown location" in error

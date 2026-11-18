"""Hostile files for `src/utils/dit_size.py`: unreadable headers fall back to the name, never raise."""
import struct

import pytest

from src.utils.dit_size import DitSize, choose_dit_size


@pytest.mark.parametrize(
    "content",
    [
        b"",                                              # zero bytes
        b"\x01\x02\x03",                                  # shorter than the length field
        struct.pack("<Q", 2**63) + b"{}",                 # absurd header length
        struct.pack("<Q", 100) + b'{"blocks.0.w": 1',     # length past the end of the file
        struct.pack("<Q", 5) + b"\xff\xfe\xfd\xfc\xfb",   # header is not UTF-8 JSON
        struct.pack("<Q", 2) + b"[]",                     # JSON, but not an object
        b"<html>not a model at all</html>" * 10,          # some other file renamed
    ],
)
@pytest.mark.parametrize("name, expected", [("x_7B_model.safetensors", DitSize.B7), ("x.safetensors", DitSize.B3)])
def test_bad_safetensors_falls_back_to_name(tmp_path, content, name, expected):
    path = tmp_path / name
    path.write_bytes(content)

    choice = choose_dit_size(name, str(path))

    assert choice.size == expected
    assert choice.source.startswith("file name")


@pytest.mark.parametrize("content", [b"", b"GGUF", b"NOPE" + b"\0" * 64])
def test_bad_gguf_falls_back_to_name(tmp_path, content):
    path = tmp_path / "model_7b.gguf"
    path.write_bytes(content)

    choice = choose_dit_size("model_7b.gguf", str(path))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_missing_file_falls_back_to_name(tmp_path):
    choice = choose_dit_size("seedvr2_ema_7b.safetensors", str(tmp_path / "absent.safetensors"))

    assert choice.size == DitSize.B7 and "FileNotFoundError" in choice.source

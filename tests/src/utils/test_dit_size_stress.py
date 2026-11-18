"""Hostile files for `src/utils/dit_size.py`: unreadable headers fall back to the name, never raise."""
import json
import os
import struct
import tracemalloc

import pytest

from src.utils.dit_size import DitSize, choose_dit_size
from test_dit_size import BLOCKS_3B, BLOCKS_7B, block_tensor_names, write_gguf, write_safetensors

# Same cap as `_MAX_SAFETENSORS_HEADER_BYTES` in dit_size.py
HEADER_CAP = 64 * 1024 * 1024
GGUF_HEADER = b"GGUF" + struct.pack("<I", 3)
GGUF_TYPE_UINT8, GGUF_TYPE_ARRAY = 0, 9


def _gguf_string(raw: bytes) -> bytes:
    return struct.pack("<Q", len(raw)) + raw


def _peak_bytes(action):
    """Peak traced Python/numpy allocation while `action` runs."""
    tracemalloc.start()
    try:
        result = action()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


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


# --- safetensors: header length and JSON ------------------------------------------------------------

@pytest.mark.parametrize(
    "content",
    [
        struct.pack("<Q", 0),                                    # zero-length header, nothing after it
        struct.pack("<Q", 0) + b"{}",                            # zero-length header, data after it
        struct.pack("<Q", 2**64 - 1) + b"{}",                    # largest u64
        struct.pack("<Q", 25) + b'{"blocks.0.w": 1}',            # length == file size (8 bytes too long)
        struct.pack("<Q", 3) + b"{}",                            # one byte past EOF
        struct.pack("<Q", 2) + b"{}\0\0",                        # empty object
        struct.pack("<Q", 4) + b"null",
        struct.pack("<Q", 2) + b'""',
        struct.pack("<Q", 2) + b"42",
        struct.pack("<Q", 28) + b'{"__metadata__": {"a": "b"}}',  # metadata only
        struct.pack("<Q", 7) + b"\xef\xbb\xbf{}\0\0",           # BOM + object + padding
        struct.pack("<Q", 6) + b"  {}  ",
        struct.pack("<Q", 9) + b'{"a": NaN',                     # truncated JSON
    ],
)
def test_odd_safetensors_headers_fall_back_to_name(tmp_path, content):
    path = tmp_path / "seedvr2_7b.safetensors"
    path.write_bytes(content)

    choice = choose_dit_size("seedvr2_7b.safetensors", str(path))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_declared_length_past_eof_does_not_allocate_it(tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(struct.pack("<Q", HEADER_CAP - 1) + b"{}")

    choice, peak = _peak_bytes(lambda: choose_dit_size("x.safetensors", str(path)))

    assert choice.source.startswith("file name")
    assert peak < 1024 * 1024


def test_declared_length_over_the_cap_in_a_big_file_is_rejected_unread(tmp_path):
    path = tmp_path / "x_7b.safetensors"
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", HEADER_CAP + 1))
        file.truncate(HEADER_CAP + 4096)  # big enough for the length to fit inside the file

    choice, peak = _peak_bytes(lambda: choose_dit_size("x_7b.safetensors", str(path)))

    assert choice.size == DitSize.B7 and "implausible" in choice.source
    assert peak < 1024 * 1024


def test_deeply_nested_json_header_falls_back(tmp_path):
    depth = 200_000
    header = b'{"blocks.0.w": ' + b"[" * depth + b"]" * depth + b"}"
    path = tmp_path / "deep_7b.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)

    choice = choose_dit_size("deep_7b.safetensors", str(path))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_block_index_with_thousands_of_digits_falls_back(tmp_path):
    names = block_tensor_names(BLOCKS_3B) + ["blocks." + "9" * 5000 + ".w"]
    path = write_safetensors(tmp_path / "huge_index_7b.safetensors", names)

    choice = choose_dit_size("huge_index_7b.safetensors", str(path))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_header_with_unicode_tensor_names_is_read(tmp_path):
    names = block_tensor_names(BLOCKS_7B) + ["модель.🎨.w", "blocks.x.y"]
    path = write_safetensors(tmp_path / "plain.safetensors", names)

    assert choose_dit_size("plain.safetensors", str(path)).size == DitSize.B7


def test_many_non_block_tensors_give_unknown_not_crash(tmp_path):
    names = [f"layer{index}.weight" for index in range(50_000)]
    path = write_safetensors(tmp_path / "other_model_7b.safetensors", names)

    choice = choose_dit_size("other_model_7b.safetensors", str(path))

    assert choice.size == DitSize.B7 and choice.source == "file name (block count not recognised)"


def test_header_keys_that_look_like_blocks_but_are_not_counted(tmp_path):
    names = ["double_blocks.40.w", "single_blocks.40.w", "xblocks.40.w"] + block_tensor_names(BLOCKS_3B)
    path = write_safetensors(tmp_path / "x.safetensors", names)

    assert choose_dit_size("x.safetensors", str(path)).size == DitSize.B3


# --- Wrong extension, directories, odd paths ---------------------------------------------------------

def test_gguf_content_named_safetensors_falls_back(tmp_path):
    real = write_gguf(tmp_path / "real.gguf", block_tensor_names(BLOCKS_3B))
    renamed = tmp_path / "renamed_7b.safetensors"
    os.replace(real, renamed)

    choice = choose_dit_size("renamed_7b.safetensors", str(renamed))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_safetensors_content_named_gguf_falls_back(tmp_path):
    path = write_safetensors(tmp_path / "renamed_7b.gguf", block_tensor_names(BLOCKS_3B))

    choice = choose_dit_size("renamed_7b.gguf", str(path))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_upper_case_gguf_extension_is_not_parsed_as_gguf(tmp_path):
    path = write_gguf(tmp_path / "m.GGUF", block_tensor_names(BLOCKS_3B))

    choice = choose_dit_size("m_7b.GGUF", str(path))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


@pytest.mark.parametrize("name", ["dir_7b.safetensors", "dir_7b.gguf"])
def test_directory_named_like_a_model_falls_back(tmp_path, name):
    (tmp_path / name).mkdir()

    choice = choose_dit_size(name, str(tmp_path / name))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


@pytest.mark.parametrize("checkpoint_path", [None, "", "\0bad.safetensors"])
def test_odd_checkpoint_paths_fall_back(checkpoint_path):
    choice = choose_dit_size("odd_7b.safetensors", checkpoint_path)

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")


def test_empty_model_name_with_unreadable_file_gives_3b(tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(b"")

    assert choose_dit_size("", str(path)).size == DitSize.B3


def test_model_name_with_folder_part_uses_only_the_file_name(tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(b"")

    assert choose_dit_size(os.path.join("models_7b", "plain.safetensors"), str(path)).size == DitSize.B3


# --- GGUF: absurd counts and lengths -----------------------------------------------------------------

@pytest.mark.parametrize(
    "content",
    [
        GGUF_HEADER + struct.pack("<QQ", 2**63, 0),                              # tensor count
        GGUF_HEADER + struct.pack("<QQ", 2**64 - 1, 0),
        GGUF_HEADER + struct.pack("<QQ", 0, 2**63),                              # key/value count
        GGUF_HEADER + struct.pack("<QQ", 0, 1) + struct.pack("<Q", 2**63) + b"key",       # key length
        GGUF_HEADER + struct.pack("<QQ", 0, 1) + struct.pack("<Q", 2**64 - 1) + b"key",
        GGUF_HEADER + struct.pack("<QQ", 1, 0) + struct.pack("<Q", 2**62) + b"blocks.0.w",  # tensor name length
        GGUF_HEADER + struct.pack("<QQ", 1, 0) + _gguf_string(b"blocks.0.w") + struct.pack("<I", 2**32 - 1),  # dims
        GGUF_HEADER + struct.pack("<QQ", 1, 0) + _gguf_string(b"\xff\xfe") + struct.pack("<IQIQ", 1, 1, 0, 0),
        GGUF_HEADER + struct.pack("<QQ", 0, 1) + _gguf_string(b"k") + struct.pack("<I", 999),  # unknown value type
        b"GGUF" + struct.pack("<I", 99) + struct.pack("<QQ", 0, 0),              # unknown version
        b"GGUF" + struct.pack("<I", 3),                                          # counts missing
    ],
)
def test_absurd_gguf_headers_fall_back_quickly(tmp_path, content):
    path = tmp_path / "model_7b.gguf"
    path.write_bytes(content)

    choice, peak = _peak_bytes(lambda: choose_dit_size("model_7b.gguf", str(path)))

    assert choice.size == DitSize.B7 and choice.source.startswith("file name")
    assert peak < 4 * 1024 * 1024


@pytest.mark.xfail(
    strict=True,
    reason="bug: gguf.GGUFReader walks a declared array length past EOF with zero-size reads, "
           "so memory and time grow with the declared length, not the file size (10**6 -> ~650 MB, ~20 s)",
)
def test_gguf_with_absurd_array_length_does_not_grow_memory(tmp_path):
    declared = 20_000  # kept small so the test stays fast; a real corrupt value would be ~2**63
    content = (
        GGUF_HEADER + struct.pack("<QQ", 0, 1) + _gguf_string(b"k")
        + struct.pack("<II", GGUF_TYPE_ARRAY, GGUF_TYPE_UINT8) + struct.pack("<Q", declared) + b"\0" * 16
    )
    path = tmp_path / "model_7b.gguf"
    path.write_bytes(content)  # 61 bytes

    choice, peak = _peak_bytes(lambda: choose_dit_size("model_7b.gguf", str(path)))

    assert choice.source.startswith("file name")
    assert peak < 4 * 1024 * 1024


# --- Files are released after reading (Windows keeps open/mapped files locked) -----------------------

def test_gguf_file_can_be_deleted_after_a_successful_read(tmp_path):
    path = write_gguf(tmp_path / "renamed.gguf", block_tensor_names(BLOCKS_7B))

    assert choose_dit_size("renamed.gguf", str(path)).source == "checkpoint header"
    os.remove(path)


def test_gguf_file_can_be_deleted_after_a_failed_read(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(GGUF_HEADER + struct.pack("<QQ", 2**63, 0))

    assert choose_dit_size("broken.gguf", str(path)).source.startswith("file name")
    os.remove(path)


def test_safetensors_file_can_be_deleted_after_a_failed_read(tmp_path):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(struct.pack("<Q", 2) + b"[]")

    assert choose_dit_size("broken.safetensors", str(path)).source.startswith("file name")
    os.remove(path)


def test_header_json_written_by_a_foreign_tool_is_read(tmp_path):
    header = {f"model.diffusion_model.blocks.{i}.w": {"dtype": "F16", "shape": [], "data_offsets": [0, 0]}
              for i in range(BLOCKS_7B)}
    raw = json.dumps(header, ensure_ascii=False, indent=2).encode() + b" " * 7  # padded like real writers
    path = tmp_path / "plain.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)

    assert choose_dit_size("plain.safetensors", str(path)).size == DitSize.B7

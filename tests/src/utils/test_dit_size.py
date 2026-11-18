"""DiT size detection from checkpoint headers (`src/utils/dit_size.py`, a local patch)."""
import json
import re
import struct
from pathlib import Path

import pytest

from src.utils.dit_size import (
    CONFIG_DIR_BY_DIT_SIZE,
    DIT_SIZE_BY_BLOCK_COUNT,
    DitSize,
    choose_dit_size,
    read_tensor_names,
    size_from_tensor_names,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

BLOCKS_3B = 32
BLOCKS_7B = 36


def block_tensor_names(block_count, prefix=""):
    names = [f"{prefix}vid_in.proj.weight"]
    for index in range(block_count):
        names += [f"{prefix}blocks.{index}.attn.proj_qkv.vid.weight", f"{prefix}blocks.{index}.mlp.vid.proj_in.weight"]
    return names


def write_safetensors(path, names):
    """Minimal valid safetensors file: one float32 per tensor."""
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]} for i, name in enumerate(names)}
    header["__metadata__"] = {"format": "pt"}
    header_bytes = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * (4 * len(names)))
    return path


def write_gguf(path, names):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), "seedvr2")
    for name in names:
        writer.add_tensor(name, np.zeros((1,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.mark.parametrize("count, size", [(BLOCKS_3B, DitSize.B3), (BLOCKS_7B, DitSize.B7)])
def test_size_from_block_count(count, size):
    assert size_from_tensor_names(block_tensor_names(count)) == size


def test_prefixed_names_are_counted():
    assert size_from_tensor_names(block_tensor_names(BLOCKS_7B, prefix="model.")) == DitSize.B7


def test_unknown_block_count_is_none():
    assert size_from_tensor_names(block_tensor_names(12)) is None
    assert size_from_tensor_names(["vae.encoder.weight"]) is None


def test_renamed_7b_safetensors_is_detected_as_7b(tmp_path):
    path = write_safetensors(tmp_path / "my_upscaler.safetensors", block_tensor_names(BLOCKS_7B))

    choice = choose_dit_size("my_upscaler.safetensors", str(path))

    assert choice.size == DitSize.B7 and choice.source == "checkpoint header"


def test_3b_named_7b_is_detected_as_3b(tmp_path):
    path = write_safetensors(tmp_path / "seedvr2_7b.safetensors", block_tensor_names(BLOCKS_3B))

    assert choose_dit_size("seedvr2_7b.safetensors", str(path)).size == DitSize.B3


def test_gguf_header_is_read(tmp_path):
    path = write_gguf(tmp_path / "renamed.gguf", block_tensor_names(BLOCKS_7B))

    assert choose_dit_size("renamed.gguf", str(path)).size == DitSize.B7


def test_metadata_key_is_not_a_tensor(tmp_path):
    path = write_safetensors(tmp_path / "m.safetensors", ["blocks.0.w"])

    assert read_tensor_names(str(path)) == ["blocks.0.w"]


@pytest.mark.parametrize("block_count, size", sorted(DIT_SIZE_BY_BLOCK_COUNT.items()))
def test_block_count_matches_num_layers_of_the_chosen_config(block_count, size):
    config = (REPO_ROOT / CONFIG_DIR_BY_DIT_SIZE[size] / "main.yaml").read_text(encoding="utf-8")

    num_layers = int(re.search(r"^\s*num_layers:\s*(\d+)", config, re.MULTILINE).group(1))

    assert num_layers == block_count

"""
Tell a 3B from a 7B SeedVR2 DiT checkpoint by its contents, not by its file name.

Local addition of the Forge Neo extension (listed in docs/external-apis/seedvr2-upstream.md,
"Local patches"): upstream chooses the model config by `"7b" in filename`, so a renamed 7B file
was built with the 3B config and failed with a shape error. Only the file header is read
(the safetensors JSON header, the GGUF tensor list), never the weights.
"""

import json
import os
import re
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional


class DitSize(str, Enum):
    B3 = "3B"
    B7 = "7B"


# Transformer block count = `num_layers` in configs_3b/main.yaml and configs_7b/main.yaml
DIT_SIZE_BY_BLOCK_COUNT = {32: DitSize.B3, 36: DitSize.B7}
CONFIG_DIR_BY_DIT_SIZE = {DitSize.B3: "./configs_3b", DitSize.B7: "./configs_7b"}

GGUF_EXTENSION = ".gguf"
_BLOCK_INDEX = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_SAFETENSORS_LENGTH_FORMAT = "<Q"  # little-endian u64 length of the JSON header
_SAFETENSORS_LENGTH_BYTES = 8
# Real DiT headers are well under 1 MB; the cap stops a corrupt length from allocating gigabytes.
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_SAFETENSORS_METADATA_KEY = "__metadata__"


@dataclass(frozen=True)
class DitSizeChoice:
    """The size to build the model with and how it was found (for the log)."""
    size: DitSize
    source: str


def choose_dit_size(dit_model: str, checkpoint_path: str) -> DitSizeChoice:
    """Size from the checkpoint header; when it cannot be read, from the file name as upstream did."""
    try:
        size = size_from_tensor_names(read_tensor_names(checkpoint_path))
    except Exception as exc:  # any unreadable or foreign file means "unknown", never a crash here
        return DitSizeChoice(_size_from_name(dit_model), f"file name (header unreadable: {type(exc).__name__}: {exc})")
    if size is None:
        return DitSizeChoice(_size_from_name(dit_model), "file name (block count not recognised)")
    return DitSizeChoice(size, "checkpoint header")


def size_from_tensor_names(names: Iterable[str]) -> Optional[DitSize]:
    """Count transformer blocks (`blocks.<N>.*` tensors); None for an unknown count."""
    indices = {int(match.group(1)) for match in map(_BLOCK_INDEX.search, names) if match}
    if not indices:
        return None
    return DIT_SIZE_BY_BLOCK_COUNT.get(max(indices) + 1)


def read_tensor_names(checkpoint_path: str) -> List[str]:
    """Tensor names from the header of a .gguf or .safetensors file."""
    if checkpoint_path.endswith(GGUF_EXTENSION):
        return _read_gguf_tensor_names(checkpoint_path)
    return _read_safetensors_tensor_names(checkpoint_path)


def _read_safetensors_tensor_names(path: str) -> List[str]:
    with open(path, "rb") as file:
        length_bytes = file.read(_SAFETENSORS_LENGTH_BYTES)
        if len(length_bytes) != _SAFETENSORS_LENGTH_BYTES:
            raise ValueError("file is shorter than a safetensors header")
        (header_length,) = struct.unpack(_SAFETENSORS_LENGTH_FORMAT, length_bytes)
        if header_length > min(_MAX_SAFETENSORS_HEADER_BYTES, os.path.getsize(path)):
            raise ValueError(f"implausible safetensors header length {header_length}")
        header = json.loads(file.read(header_length))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return [name for name in header if name != _SAFETENSORS_METADATA_KEY]


def _read_gguf_tensor_names(path: str) -> List[str]:
    import gguf  # Forge Neo ships it; imported lazily so safetensors-only setups never need it

    reader = gguf.GGUFReader(path)
    try:
        return [str(tensor.name) for tensor in reader.tensors]
    finally:
        # Drops our reference to the memory map; on an exception the traceback would otherwise keep
        # it alive and the file stays locked on Windows.
        del reader


def _size_from_name(dit_model: str) -> DitSize:
    return DitSize.B7 if "7b" in os.path.basename(dit_model).lower() else DitSize.B3

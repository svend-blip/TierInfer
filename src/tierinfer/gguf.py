"""Reading a GGUF file's directory without reading its weights.

A GGUF file is a header, a table of typed metadata, a directory of tensors,
and then the tensor data itself. The directory is small — a few hundred
kilobytes for a 56 GB model — so the whole layout can be known without
touching the weights. That is what makes byte-range loading possible at all,
and it is the first thing TierInfer needs.

Measured against GLM-4.5-Air-Derestricted IQ4_XS (glm4moe, GGUF v3): 803
tensors and 53 metadata keys, read in well under a second from a 56.5 GB
file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# GGML type id -> (name, elements per block, bytes per block).
# A quantized tensor stores whole blocks, so its byte size is
# (elements / block_elements) * block_bytes. Types absent here are not
# supported for size arithmetic; `ggml_type_name` still names them.
_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


class GGUFError(Exception):
    """The file is not a GGUF file, or its directory cannot be read."""


def ggml_type_name(type_id: int) -> str:
    """The type's name, or ``TYPE_<id>`` when this build does not know it."""
    known = _TYPES.get(type_id)
    return known[0] if known else f"TYPE_{type_id}"


def tensor_nbytes(type_id: int, dims: tuple[int, ...]) -> int:
    """Bytes a tensor of this type and shape occupies in the data section.

    Raises for a type whose block layout this module does not carry, rather
    than guessing: a wrong size here would produce byte ranges that read the
    wrong weights, which is worse than refusing to answer.
    """
    known = _TYPES.get(type_id)
    if known is None:
        raise GGUFError(f"no block layout for ggml type {type_id}")
    _, block_elements, block_bytes = known
    elements = 1
    for d in dims:
        elements *= d
    if elements % block_elements:
        raise GGUFError(
            f"{elements} elements is not a whole number of "
            f"{block_elements}-element {known[0]} blocks"
        )
    return (elements // block_elements) * block_bytes


@dataclass(frozen=True)
class TensorEntry:
    """One row of the tensor directory, with its absolute file offset.

    ``offset`` as stored in the file is relative to the start of the data
    section; ``file_offset`` is what a reader actually seeks to.
    """

    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int
    file_offset: int
    nbytes: int

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.type_id)


@dataclass(frozen=True)
class GGUFFile:
    """A GGUF file's directory: its metadata and where every tensor lives."""

    path: Path
    version: int
    alignment: int
    data_offset: int
    metadata: dict[str, Any]
    tensors: tuple[TensorEntry, ...]

    def tensor(self, name: str) -> TensorEntry:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_key(self, suffix: str, default: Any = None) -> Any:
        """Read ``<architecture>.<suffix>``, the GGUF convention for per-model keys."""
        return self.metadata.get(f"{self.architecture}.{suffix}", default)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFError(f"file ended while reading {n} bytes")
    return data


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f

    def u32(self) -> int:
        return struct.unpack("<I", _read_exact(self.f, 4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", _read_exact(self.f, 8))[0]

    def string(self) -> str:
        return _read_exact(self.f, self.u64()).decode("utf-8", "replace")

    def value(self, type_id: int) -> Any:
        f = self.f
        if type_id == 0:
            return struct.unpack("<B", _read_exact(f, 1))[0]
        if type_id == 1:
            return struct.unpack("<b", _read_exact(f, 1))[0]
        if type_id == 2:
            return struct.unpack("<H", _read_exact(f, 2))[0]
        if type_id == 3:
            return struct.unpack("<h", _read_exact(f, 2))[0]
        if type_id == 4:
            return self.u32()
        if type_id == 5:
            return struct.unpack("<i", _read_exact(f, 4))[0]
        if type_id == 6:
            return struct.unpack("<f", _read_exact(f, 4))[0]
        if type_id == 7:
            return struct.unpack("<?", _read_exact(f, 1))[0]
        if type_id == 8:
            return self.string()
        if type_id == 9:
            element_type = self.u32()
            count = self.u64()
            return [self.value(element_type) for _ in range(count)]
        if type_id == 10:
            return self.u64()
        if type_id == 11:
            return struct.unpack("<q", _read_exact(f, 8))[0]
        if type_id == 12:
            return struct.unpack("<d", _read_exact(f, 8))[0]
        raise GGUFError(f"unknown metadata value type {type_id}")


def read_gguf(path: str | Path) -> GGUFFile:
    """Read the directory of a GGUF file. The weights are never touched."""
    path = Path(path)
    with path.open("rb") as f:
        r = _Reader(f)
        if _read_exact(f, 4) != GGUF_MAGIC:
            raise GGUFError(f"{path} does not start with the GGUF magic")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = r.u64()
        kv_count = r.u64()

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = r.string()
            metadata[key] = r.value(r.u32())

        raw: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = tuple(r.u64() for _ in range(n_dims))
            type_id = r.u32()
            offset = r.u64()
            raw.append((name, dims, type_id, offset))

        alignment = int(metadata.get("general.alignment", 32))
        here = f.tell()
        data_offset = here if here % alignment == 0 else here + (alignment - here % alignment)

    tensors = tuple(
        TensorEntry(
            name=name,
            dims=dims,
            type_id=type_id,
            offset=offset,
            file_offset=data_offset + offset,
            nbytes=tensor_nbytes(type_id, dims),
        )
        for name, dims, type_id, offset in raw
    )
    return GGUFFile(
        path=path,
        version=version,
        alignment=alignment,
        data_offset=data_offset,
        metadata=metadata,
        tensors=tensors,
    )

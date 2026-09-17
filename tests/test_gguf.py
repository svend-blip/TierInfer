"""Tests for the GGUF directory reader, against a file this test builds itself.

No fixture points at a real model: a 56 GB file is not a test dependency, and
a test that needs one is a test that gets skipped. The synthetic file carries
the same shapes the real one does — a fused expert tensor with the expert
index last — so the arithmetic under test is the arithmetic that runs.
"""

from __future__ import annotations

import struct

import pytest

from tierinfer.gguf import GGUFError, read_gguf, tensor_nbytes, ggml_type_name

IQ4_XS = 23
F32 = 0


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, tensors, metadata=None, alignment=32):
    """Write a GGUF v3 file whose tensor data is zeros of the right size."""
    metadata = dict(metadata or {})
    metadata.setdefault("general.alignment", alignment)
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        head += _s(key)
        if isinstance(value, str):
            head += struct.pack("<I", 8) + _s(value)
        else:
            head += struct.pack("<I", 4) + struct.pack("<I", int(value))
    offset = 0
    sizes = []
    for name, dims, type_id in tensors:
        head += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            head += struct.pack("<Q", d)
        head += struct.pack("<II", type_id, 0)[:4] + struct.pack("<Q", offset)
        n = tensor_nbytes(type_id, dims)
        sizes.append(n)
        offset += n
    pad = (-len(head)) % alignment
    path.write_bytes(bytes(head) + b"\0" * pad + b"\0" * offset)
    return path


def test_the_directory_is_read_without_touching_the_weights(tmp_path):
    p = write_gguf(tmp_path / "m.gguf",
                   [("blk.0.ffn_gate_exps.weight", (256, 4, 8), IQ4_XS)],
                   {"general.architecture": "glm4moe"})
    g = read_gguf(p)
    assert g.architecture == "glm4moe"
    assert len(g.tensors) == 1
    t = g.tensors[0]
    assert t.file_offset == g.data_offset
    assert t.nbytes == tensor_nbytes(IQ4_XS, (256, 4, 8))


def test_every_tensor_ends_exactly_where_the_file_does(tmp_path):
    """The check that caught nothing on the real model, which is why it is here."""
    p = write_gguf(tmp_path / "m.gguf", [
        ("token_embd.weight", (256, 8), F32),
        ("blk.0.ffn_down_exps.weight", (256, 4, 8), IQ4_XS),
    ])
    g = read_gguf(p)
    last = max(g.tensors, key=lambda t: t.file_offset)
    assert last.file_offset + last.nbytes == p.stat().st_size


def test_a_type_without_a_block_layout_refuses_rather_than_guesses():
    with pytest.raises(GGUFError):
        tensor_nbytes(999, (256,))
    assert ggml_type_name(999) == "TYPE_999"


def test_a_shape_that_is_not_whole_blocks_refuses():
    with pytest.raises(GGUFError):
        tensor_nbytes(IQ4_XS, (100,))


def test_a_file_that_is_not_gguf_is_refused(tmp_path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(GGUFError):
        read_gguf(p)

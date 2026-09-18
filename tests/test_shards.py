"""A split model is several files, and every byte range must say which.

`gguf-split` writes `<stem>-00001-of-0000N.gguf` and puts `split.no`,
`split.count` and `split.tensors.count` in each shard. The 480B validation
model is six such files. Before this, `read_gguf` on the first shard produced
an index of eleven layers out of sixty-two and every budget built on it was
wrong by construction — so these tests build a split, read it whole, and
read bytes across the seam.
"""

from __future__ import annotations

import os

import pytest

from tierinfer.gguf import GGUFError, read_gguf, read_model, shard_paths
from tierinfer.index import ModelIndex, load
from tierinfer.storage import StorageBackend
from tierinfer.stream import BufferPool, ExpertStreamer

from test_gguf import IQ4_XS, F32, write_gguf

EXPERTS = 4


def _layer(l, experts=EXPERTS):
    return [
        (f"blk.{l}.attn_q.weight", (256, 8), IQ4_XS),
        (f"blk.{l}.ffn_gate_inp.weight", (256, experts), F32),
        (f"blk.{l}.ffn_gate_exps.weight", (256, 4, experts), IQ4_XS),
        (f"blk.{l}.ffn_up_exps.weight", (256, 4, experts), IQ4_XS),
        (f"blk.{l}.ffn_down_exps.weight", (256, 4, experts), IQ4_XS),
    ]


def _fill(path, seed):
    """Overwrite the data section with a pattern unique to this shard."""
    g = read_gguf(path)
    raw = bytearray(path.read_bytes())
    for i in range(g.data_offset, len(raw)):
        raw[i] = (i * 7 + seed) % 251
    path.write_bytes(bytes(raw))


def split_model(tmp_path, count=3, layers_per=2):
    """A split of ``count`` shards, each holding ``layers_per`` layers."""
    total = 0
    paths = []
    for no in range(count):
        tensors = [("token_embd.weight", (256, 8), F32)] if no == 0 else []
        for l in range(no * layers_per, (no + 1) * layers_per):
            tensors += _layer(l)
        total += len(tensors)
        paths.append((no, tensors))
    out = []
    for no, tensors in paths:
        meta = {"split.no": no, "split.count": count, "split.tensors.count": total}
        if no == 0:
            meta.update({"general.architecture": "qwen3moe",
                         "qwen3moe.expert_count": EXPERTS,
                         "qwen3moe.expert_used_count": 2,
                         "qwen3moe.block_count": count * layers_per})
        p = write_gguf(tmp_path / f"m-{no + 1:05d}-of-{count:05d}.gguf", tensors, meta)
        _fill(p, seed=no + 1)
        out.append(p)
    return out


# -- discovery ----------------------------------------------------------


def test_a_single_file_is_its_own_only_shard(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [("token_embd.weight", (256, 8), F32)],
                   {"general.architecture": "glm4moe"})
    assert shard_paths(p) == [p]
    g = read_model(p)
    assert g.shards == (p,)
    assert g.files == (p,)
    assert all(t.path == p for t in g.tensors)


def test_the_shards_are_found_from_any_one_of_them(tmp_path):
    shards = split_model(tmp_path)
    assert shard_paths(shards[1]) == shards
    assert shard_paths(shards[2]) == shards


def test_a_split_reads_as_one_model(tmp_path):
    shards = split_model(tmp_path, count=3, layers_per=2)
    g = read_model(shards[0])
    assert g.shards == tuple(shards)
    assert g.architecture == "qwen3moe"
    assert len(g.tensors) == 1 + 6 * 5
    names = {t.name for t in g.tensors}
    assert "blk.0.attn_q.weight" in names and "blk.5.ffn_down_exps.weight" in names
    # each tensor knows its file, and its offset is relative to that file
    for t in g.tensors:
        assert t.path in shards
        assert t.file_offset + t.nbytes <= t.path.stat().st_size
    assert g.nbytes_on_disk == sum(p.stat().st_size for p in shards)


def test_the_index_sees_every_layer_of_a_split(tmp_path):
    shards = split_model(tmp_path, count=3, layers_per=2)
    ix = load(shards[0])
    assert ix.moe_layers == [0, 1, 2, 3, 4, 5]
    ref = ix.expert(5, 3)
    assert {r.path for r in ref.ranges} == {shards[2]}
    # the working set counts all six layers, not the first shard's two
    assert ix.working_set_nbytes() == ix.always_resident_nbytes() + 6 * 2 * ix.expert_nbytes()


# -- refusals -----------------------------------------------------------


def test_a_missing_shard_is_refused_not_narrowed(tmp_path):
    shards = split_model(tmp_path)
    os.unlink(shards[1])
    with pytest.raises(GGUFError, match="missing 1 of 3"):
        shard_paths(shards[0])


def test_a_shard_out_of_order_is_refused(tmp_path):
    shards = split_model(tmp_path)
    # swap the contents of shards 2 and 3 so the names lie about split.no
    a, b = shards[1].read_bytes(), shards[2].read_bytes()
    shards[1].write_bytes(b)
    shards[2].write_bytes(a)
    with pytest.raises(GGUFError, match="split.no"):
        read_model(shards[0])


def test_a_split_file_with_an_unconventional_name_is_refused(tmp_path):
    shards = split_model(tmp_path)
    odd = tmp_path / "renamed.gguf"
    shards[0].rename(odd)
    with pytest.raises(GGUFError, match="not named"):
        shard_paths(odd)


# -- reading across the seam --------------------------------------------


def _expected(ref):
    out = b""
    for r in ref.ranges:
        with open(r.path, "rb") as fh:
            fh.seek(r.file_offset)
            out += fh.read(r.nbytes)
    return out


def test_the_backend_reads_each_range_from_its_own_shard(tmp_path):
    shards = split_model(tmp_path)
    ix = load(shards[0])
    with StorageBackend.for_model(ix.gguf) as b:
        assert b.paths == tuple(shards)
        for layer in ix.moe_layers:
            for e in range(EXPERTS):
                ref = ix.expert(layer, e)
                blobs, _ = b.read(list(ref.ranges))
                assert b"".join(blobs) == _expected(ref), (layer, e)


def test_the_streamer_reads_across_shards(tmp_path):
    shards = split_model(tmp_path)
    ix = load(shards[0])
    refs = [ix.expert(l, e) for l in ix.moe_layers for e in range(EXPERTS)]
    slot = max(r.nbytes for r in refs)
    with StorageBackend.for_model(ix.gguf) as b, \
            ExpertStreamer(b, BufferPool(slot, len(refs)), workers=3) as s:
        loads = [s.submit((r.layer, r.expert), r.ranges) for r in refs]
        for ref, load_ in zip(refs, loads):
            assert bytes(s.wait(load_, timeout=10)) == _expected(ref)
            s.release(load_)


def test_a_range_naming_a_file_the_backend_did_not_open_is_an_error(tmp_path):
    shards = split_model(tmp_path)
    ix = load(shards[0])
    ref = ix.expert(4, 0)                       # lives in shard 3
    with StorageBackend(shards[0]) as b:        # opened only shard 1
        with pytest.raises(OSError, match="did not open"):
            b.read(list(ref.ranges))


def test_a_single_file_backend_still_accepts_pathless_ranges(tmp_path):
    from tierinfer.index import ByteRange
    p = tmp_path / "blob"
    p.write_bytes(bytes(range(256)) * 16)
    with StorageBackend(p) as b:
        blobs, _ = b.read([ByteRange("x", 256, 16)])
    assert blobs[0] == bytes(range(16))

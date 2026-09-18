"""The FTW layout inspector, on a checkpoint the test writes: two shards,
banks whose rows are experts, a range that crosses the shard boundary."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.ftw import FTWError, FTWIndex  # noqa: E402

ALIGN = 4096
EXPERTS = 4


def write_ftw(dirpath, shard_limit=ALIGN * 12):
    """Entries: a dense weight, then per layer two banks; shards cut at shard_limit."""
    entries = []
    off = 0

    def add(name, kind, nbytes, shape):
        nonlocal off
        entries.append({"name": name, "kind": kind, "dtype": "uint8", "shape": shape,
                        "global_off": off, "nbytes": nbytes})
        off += -(-nbytes // ALIGN) * ALIGN

    add("model.norm.weight", "weight", 2048, [1024])
    for l in range(2):
        add(f"gate_up_packed#L{l:05d}", "experts_bank", EXPERTS * ALIGN * 2, [EXPERTS, 8192])
        add(f"down_packed#L{l:05d}", "experts_bank", EXPERTS * ALIGN, [EXPERTS, 4096])
    total = off
    shards = []
    cur = 0
    i = 0
    data = bytes((b * 7 + 3) % 251 for b in range(total))
    while cur < total:
        n = min(shard_limit, total - cur)
        name = f"freetoken-{i:05d}.ftw"
        (dirpath / name).write_bytes(data[cur:cur + n])
        shards.append({"file": name, "global_off": cur, "nbytes": n})
        cur += n
        i += 1
    (dirpath / "freetoken_weight.json").write_text(json.dumps({
        "format": "freetoken_weight", "version": 1, "align": ALIGN, "shard_limit": shard_limit,
        "total_bytes": total, "tensors": entries, "shards": shards}))
    return data


def test_the_index_is_read_and_experts_are_rows(tmp_path):
    data = write_ftw(tmp_path)
    ix = FTWIndex(tmp_path)
    assert ix.moe_layers == [0, 1] and ix.expert_count == EXPERTS
    assert len(ix.files) >= 2
    rows = ix.expert_rows(1, 2)
    assert {r.name.split("#expert")[0] for r in rows} == {"down_packed#L00001", "gate_up_packed#L00001"}
    # the bytes behind the ranges are the logical region's bytes for that row
    for r in rows:
        with open(r.path, "rb") as fh:
            fh.seek(r.file_offset)
            got = fh.read(r.nbytes)
        entry = next(e for e in ix.entries if r.name.startswith(e.name))
        logical = entry.global_off + 2 * (entry.nbytes // EXPERTS)
        shard = ix.shard_for(logical)
        assert got == data[shard.global_off + r.file_offset: shard.global_off + r.file_offset + r.nbytes]
    assert ix.expert_nbytes() == ALIGN * 3


def test_a_range_crossing_the_shard_boundary_is_split_and_still_correct(tmp_path):
    data = write_ftw(tmp_path, shard_limit=ALIGN * 5)
    ix = FTWIndex(tmp_path)
    crossing = [e for e in ix.entries if any(e.global_off < s.end < e.end for s in ix.shards)]
    assert crossing, "the fixture must place an entry across a shard boundary"
    e = crossing[0]
    pieces = ix.physical(e.name, e.global_off, e.nbytes)
    assert len(pieces) >= 2
    assert sum(p.nbytes for p in pieces) == e.nbytes
    joined = b"".join(open(p.path, "rb").read()[p.file_offset:p.file_offset + p.nbytes] for p in pieces)
    assert joined == data[e.global_off:e.end]


def test_a_byte_names_its_expert_or_its_floor(tmp_path):
    write_ftw(tmp_path)
    ix = FTWIndex(tmp_path)
    key, entry = ix.logical_to_key(0)
    assert key == ("floor", "model.norm.weight")
    e = next(x for x in ix.entries if x.name == "down_packed#L00001")
    key, _ = ix.logical_to_key(e.global_off + 3 * (e.nbytes // EXPERTS) + 5)
    assert key == (1, 3)
    assert ix.logical_to_key(10 ** 12) is None


def test_refusals(tmp_path):
    with pytest.raises(FTWError, match="not an FTW"):
        FTWIndex(tmp_path)
    write_ftw(tmp_path)
    ix = FTWIndex(tmp_path)
    with pytest.raises(FTWError):
        ix.expert_rows(0, EXPERTS)
    with pytest.raises(FTWError):
        ix.expert_rows(7, 0)
    os.unlink(tmp_path / "freetoken-00001.ftw")
    with pytest.raises(FTWError, match="missing"):
        FTWIndex(tmp_path)

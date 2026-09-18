"""The FreeToken Weight (FTW) checkpoint as a model layout.

FreeToken's own on-disk format (`freetoken/checkpoint/ftw.py`) is one
logical byte region holding every tensor at a 4096-aligned offset, sliced
physically into shard files, with an index naming each tensor's kind,
dtype, shape, logical offset and size. Expert banks are entries of kind
``experts_bank`` named ``<bank>#L<layer>`` whose first dimension is the
expert: row ``e`` of ``gate_up_packed#L00012`` is expert 12/e's packed
gate-up weights, and its byte range is arithmetic on the entry — the same
shape of fact `tierinfer.index` establishes for a GGUF's fused tensors.

This module reads the index and answers the two questions the loader
asks of any layout: which (shard file, offset, length) holds a logical
range, and which expert — or which floor tensor — a byte belongs to. It
does not parse tensors and it does not guess: an entry that does not
divide into ``num_experts`` whole rows, or a logical range that falls
outside every shard, is an error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .index import ByteRange

INDEX_NAME = "freetoken_weight.json"
_BANK = re.compile(r"^(?P<bank>.+)#L(?P<layer>\d+)$")


class FTWError(ValueError):
    pass


@dataclass(frozen=True)
class FTWEntry:
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    global_off: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.global_off + self.nbytes


@dataclass(frozen=True)
class FTWShard:
    path: Path
    global_off: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.global_off + self.nbytes


class FTWIndex:
    """One FTW checkpoint directory, addressable by logical offset and by expert."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        index = self.directory / INDEX_NAME
        if not index.exists():
            raise FTWError(f"{self.directory} has no {INDEX_NAME}; not an FTW checkpoint")
        doc = json.loads(index.read_text())
        if doc.get("format") != "freetoken_weight":
            raise FTWError(f"{index}: format {doc.get('format')!r} is not freetoken_weight")
        self.version = int(doc.get("version", 0))
        self.align = int(doc.get("align", 4096))
        self.total_bytes = int(doc.get("total_bytes", 0))
        self.shards = tuple(FTWShard(self.directory / s["file"], int(s["global_off"]), int(s["nbytes"]))
                            for s in doc["shards"])
        self.entries = tuple(FTWEntry(t["name"], t["kind"], t.get("dtype", ""), tuple(int(x) for x in t["shape"]),
                                      int(t["global_off"]), int(t["nbytes"])) for t in doc["tensors"])
        for s in self.shards:
            if not s.path.exists():
                raise FTWError(f"shard {s.path.name} named by the index is missing")
            if s.path.stat().st_size != s.nbytes:
                raise FTWError(f"shard {s.path.name} is {s.path.stat().st_size} bytes, index says {s.nbytes}")
        self.banks: dict[int, dict[str, FTWEntry]] = {}
        for e in self.entries:
            m = _BANK.match(e.name)
            if e.kind == "experts_bank" and m:
                self.banks.setdefault(int(m.group("layer")), {})[m.group("bank")] = e
        self.num_experts = 0
        for layer in self.banks.values():
            for e in layer.values():
                if e.shape:
                    self.num_experts = e.shape[0] if not self.num_experts else self.num_experts
                    if e.shape[0] != self.num_experts:
                        raise FTWError(f"{e.name} has {e.shape[0]} rows; other banks have {self.num_experts}")

    # -- questions -----------------------------------------------------------

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(s.path for s in self.shards)

    @property
    def moe_layers(self) -> list[int]:
        return sorted(self.banks)

    @property
    def expert_count(self) -> int:
        return self.num_experts

    def shard_for(self, global_off: int) -> FTWShard:
        for s in self.shards:
            if s.global_off <= global_off < s.end:
                return s
        raise FTWError(f"logical offset {global_off} falls outside every shard")

    def physical(self, name: str, global_off: int, nbytes: int) -> list[ByteRange]:
        """The (shard, offset, length) pieces of a logical range. A range may
        cross a shard boundary; both sides stay 4096-aligned by the format."""
        out: list[ByteRange] = []
        cur, end = global_off, global_off + nbytes
        while cur < end:
            s = self.shard_for(cur)
            n = min(end, s.end) - cur
            out.append(ByteRange(name=name, file_offset=cur - s.global_off, nbytes=n, path=s.path))
            cur += n
        return out

    def expert_rows(self, layer: int, expert: int) -> list[ByteRange]:
        """Every bank's row for one expert of one layer, as physical ranges."""
        banks = self.banks.get(layer)
        if not banks:
            raise FTWError(f"layer {layer} has no expert banks")
        if not 0 <= expert < self.num_experts:
            raise FTWError(f"expert {expert} outside 0..{self.num_experts - 1}")
        out: list[ByteRange] = []
        for bank, e in sorted(banks.items()):
            if e.nbytes % self.num_experts:
                raise FTWError(f"{e.name} is {e.nbytes} bytes, not {self.num_experts} whole rows")
            row = e.nbytes // self.num_experts
            out += self.physical(f"{e.name}#expert{expert}", e.global_off + expert * row, row)
        return out

    def expert_nbytes(self, layer: int | None = None) -> int:
        layer = self.moe_layers[0] if layer is None else layer
        return sum(r.nbytes for r in self.expert_rows(layer, 0))

    def floor_entries(self) -> list[FTWEntry]:
        """Everything that is not an expert bank: dense weights, scales, router."""
        bank_names = {e.name for layer in self.banks.values() for e in layer.values()}
        return [e for e in self.entries if e.name not in bank_names]

    def logical_to_key(self, global_off: int) -> tuple[object, FTWEntry] | None:
        """Which expert (layer, e) or which floor entry a logical byte is in."""
        for e in self.entries:
            if e.global_off <= global_off < e.end:
                m = _BANK.match(e.name)
                if e.kind == "experts_bank" and m and self.num_experts:
                    row = e.nbytes // self.num_experts
                    return (int(m.group("layer")), (global_off - e.global_off) // row), e
                return ("floor", e.name), e
        return None

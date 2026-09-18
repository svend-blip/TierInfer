"""Turning a GGUF directory into logical objects TierInfer can ask for.

The point of this module is one sentence from the scope: a reader should be
able to say *load expert 37 of layer 18* and get a byte range, rather than
touching a page and hoping the right weights are behind it.

How experts are stored decides whether that is possible, and it is not the
same in every model. In GLM-4.5-Air (``glm4moe``) the routed experts of a
layer are **fused into one tensor per projection**: ``ffn_gate_exps``,
``ffn_up_exps`` and ``ffn_down_exps``, each with the expert index as its last
dimension. Measured on the IQ4_XS build: ``blk.10.ffn_down_exps.weight`` has
dims (1408, 4096, 128) and occupies 396.0 MB, which divides evenly into 128
slabs of 3 244 032 bytes.

So an expert is a slice inside a tensor, not a tensor of its own, and its
byte range is arithmetic on the fused tensor's offset. This module does that
arithmetic and refuses rather than guesses when a tensor does not divide
evenly — a wrong range reads the wrong weights, which is worse than no range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .gguf import GGUFFile, GGUFError, TensorEntry, read_gguf, read_model

_BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")

#: Suffixes of the fused routed-expert tensors, in the order a forward pass
#: needs them. Named per projection because a single expert is all three.
EXPERT_PROJECTIONS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


@dataclass(frozen=True)
class ByteRange:
    """A contiguous region of the model file, and what it holds."""

    name: str
    file_offset: int
    nbytes: int
    #: Which file ``file_offset`` indexes. ``None`` means the backend's only
    #: file — the single-file case, and what tests that build ranges by hand
    #: get. A split model always names the shard.
    path: Path | None = None

    @property
    def end(self) -> int:
        return self.file_offset + self.nbytes


@dataclass(frozen=True)
class ExpertRef:
    """One routed expert of one layer: the slices that together are its weights."""

    layer: int
    expert: int
    ranges: tuple[ByteRange, ...]

    @property
    def nbytes(self) -> int:
        return sum(r.nbytes for r in self.ranges)


@dataclass(frozen=True)
class LayerLayout:
    """What one transformer block holds, split by how TierInfer must treat it."""

    layer: int
    attention: tuple[TensorEntry, ...]
    norms: tuple[TensorEntry, ...]
    router: tuple[TensorEntry, ...]
    shared_expert: tuple[TensorEntry, ...]
    routed_experts: tuple[TensorEntry, ...]

    @property
    def is_moe(self) -> bool:
        return bool(self.routed_experts)

    def resident_nbytes(self) -> int:
        """Bytes that must be resident for the layer regardless of routing."""
        return sum(
            t.nbytes
            for group in (self.attention, self.norms, self.router, self.shared_expert)
            for t in group
        )

    def routed_nbytes(self) -> int:
        return sum(t.nbytes for t in self.routed_experts)


class ModelIndex:
    """The layout of one model, addressable as logical objects."""

    def __init__(self, gguf: GGUFFile):
        self.gguf = gguf
        self.expert_count = int(gguf.arch_key("expert_count", 0) or 0)
        self.expert_used_count = int(gguf.arch_key("expert_used_count", 0) or 0)
        self.block_count = int(gguf.arch_key("block_count", 0) or 0)
        self.layers: dict[int, LayerLayout] = {}
        self._global: list[TensorEntry] = []
        self._classify()

    # -- construction ---------------------------------------------------

    def _classify(self) -> None:
        buckets: dict[int, dict[str, list[TensorEntry]]] = {}
        for t in self.gguf.tensors:
            m = _BLOCK.match(t.name)
            if not m:
                self._global.append(t)
                continue
            layer = int(m.group(1))
            leaf = m.group(2)
            b = buckets.setdefault(layer, {k: [] for k in
                                           ("attention", "norms", "router", "shared", "routed")})
            b[self._bucket_for(leaf)].append(t)
        for layer, b in sorted(buckets.items()):
            self.layers[layer] = LayerLayout(
                layer=layer,
                attention=tuple(b["attention"]),
                norms=tuple(b["norms"]),
                router=tuple(b["router"]),
                shared_expert=tuple(b["shared"]),
                routed_experts=tuple(b["routed"]),
            )

    @staticmethod
    def _bucket_for(leaf: str) -> str:
        if leaf.endswith("_exps.weight"):
            return "routed"
        if "shexp" in leaf:
            return "shared"
        if "gate_inp" in leaf or "exp_probs" in leaf:
            return "router"
        if "norm" in leaf:
            return "norms"
        if leaf.startswith("attn"):
            return "attention"
        return "shared"  # dense FFN of a non-MoE block: always resident

    # -- queries --------------------------------------------------------

    @property
    def moe_layers(self) -> list[int]:
        return [n for n, l in sorted(self.layers.items()) if l.is_moe]

    def expert(self, layer: int, expert: int) -> ExpertRef:
        """The byte ranges of one routed expert.

        Raises ``GGUFError`` when the layer has no routed experts, when the
        index is out of range, or when a fused tensor does not divide evenly
        into ``expert_count`` slabs.
        """
        layout = self.layers.get(layer)
        if layout is None or not layout.is_moe:
            raise GGUFError(f"layer {layer} has no routed experts")
        if self.expert_count <= 0:
            raise GGUFError("model metadata does not state an expert count")
        if not 0 <= expert < self.expert_count:
            raise GGUFError(f"expert {expert} outside 0..{self.expert_count - 1}")

        ranges: list[ByteRange] = []
        for suffix in EXPERT_PROJECTIONS:
            name = f"blk.{layer}.{suffix}"
            try:
                t = self.gguf.tensor(name)
            except KeyError:
                continue
            if t.dims[-1] != self.expert_count:
                raise GGUFError(
                    f"{name} last dimension {t.dims[-1]} is not the expert count "
                    f"{self.expert_count}; this layout is not sliceable by expert"
                )
            if t.nbytes % self.expert_count:
                raise GGUFError(
                    f"{name} is {t.nbytes} bytes, which is not {self.expert_count} whole slabs"
                )
            slab = t.nbytes // self.expert_count
            ranges.append(ByteRange(
                name=f"{name}#expert{expert}",
                file_offset=t.file_offset + expert * slab,
                nbytes=slab,
                path=t.path,
            ))
        if not ranges:
            raise GGUFError(f"layer {layer} has no fused expert tensors")
        return ExpertRef(layer=layer, expert=expert, ranges=tuple(ranges))

    def expert_nbytes(self) -> int:
        """Bytes one routed expert occupies, taken from the first MoE layer.

        Not every layer's expert is the same size: a Q4_K_M build quantises
        some layers' down projection to Q6_K and others to Q4_K, so on the
        480B validation model experts run from 26.5 to 30.6 MB. Buffer slots
        and budgets should use :meth:`expert_nbytes_max`; this stays as the
        single representative figure callers already rely on.
        """
        moe = self.moe_layers
        if not moe:
            return 0
        return self.expert(moe[0], 0).nbytes

    def expert_nbytes_by_layer(self) -> dict[int, int]:
        """Bytes of one expert in each MoE layer (expert 0; a layer's slabs are equal)."""
        return {layer: self.expert(layer, 0).nbytes for layer in self.moe_layers}

    def expert_nbytes_max(self) -> int:
        """The largest expert in the model: what a fixed slot has to hold."""
        sizes = self.expert_nbytes_by_layer()
        return max(sizes.values()) if sizes else 0

    def always_resident_nbytes(self) -> int:
        """Everything that is not a routed expert: the floor under any budget."""
        return sum(t.nbytes for t in self._global) + sum(
            l.resident_nbytes() for l in self.layers.values()
        )

    def routed_nbytes(self) -> int:
        return sum(l.routed_nbytes() for l in self.layers.values())

    def working_set_nbytes(self, experts_per_layer: int | None = None) -> int:
        """Bytes needed for one token: the resident floor plus the routed experts it uses.

        This is the number the whole project turns on — the working set rather
        than the total parameter count.
        """
        k = self.expert_used_count if experts_per_layer is None else experts_per_layer
        return self.always_resident_nbytes() + len(self.moe_layers) * k * self.expert_nbytes()


def load(path: str | Path) -> ModelIndex:
    """Index a model from any one of its files. A split is read whole."""
    return ModelIndex(read_model(path))

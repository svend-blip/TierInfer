"""Reading a real routing trace, and saying what it is made of.

`tools/trace/tierinfer-trace` writes one JSONL line per decode per MoE layer:

    {"decode":0,"layer":1,"n_tokens":6,"n_used":8,"experts":[[...],[...]]}

A decode is one `llama_decode` call, so decode 0 carries every prompt token
at once and each later decode carries the single token just generated. That
distinction matters and is kept rather than flattened: prompt tokens are
processed in parallel with full attention over each other, generated tokens
one at a time, and a predictor that looks strong on one may be useless on the
other.

**Routings are ragged, and legitimately so.** llama.cpp computes the final
layer only for tokens whose logits are wanted — "return the output only for
the last token", `llama-batch.cpp` — so in a six-token prompt decode, layers
1..44 carry six rows and layer 45 carries one, belonging to the last token.
A layer with fewer rows than the decode covers the decode's *last* tokens.
Tokens that layer skipped simply do not have it, and a predictor scored on
them must see a missing layer rather than an empty routing: those are
different claims.

The reader is strict about everything else. A layer missing from a decode
entirely, or a row count that is neither the full width nor a suffix of it,
is a broken trace, and a number computed from one would mean nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class TokenRouting:
    """One token's routing through every MoE layer."""

    decode: int
    index: int          # position within the decode
    prompt: bool        # part of the prompt, or generated
    routing: dict[int, tuple[int, ...]]

    def as_mapping(self) -> dict[int, Sequence[int]]:
        """The shape the predictors take."""
        return {l: list(e) for l, e in self.routing.items()}


@dataclass
class TraceInfo:
    path: str
    decodes: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    layers: tuple[int, ...] = ()
    n_used: int = 0
    experts_seen: int = 0
    #: Tokens that some layer skipped — the final layer runs only for outputs.
    partial_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


def read_trace(path: str | Path, *, prompt: bool = True,
               generated: bool = True) -> list[TokenRouting]:
    """Assemble per-token routings from a trace file.

    ``prompt`` and ``generated`` select which tokens come back. Scoring a
    prefetcher on prompt tokens alone flatters it — they arrive in one batch,
    so every expert of every layer is needed at once and nothing is being
    predicted ahead. Generation is where prediction has a job.
    """
    by_decode: dict[int, dict[int, list[list[int]]]] = defaultdict(dict)
    n_used_seen: set[int] = set()

    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                d, layer = int(rec["decode"]), int(rec["layer"])
                experts = rec["experts"]
                n_tokens, n_used = int(rec["n_tokens"]), int(rec["n_used"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise TraceError(f"{path}:{lineno}: not a trace record ({e})") from None
            if len(experts) != n_tokens:
                raise TraceError(f"{path}:{lineno}: claims {n_tokens} tokens, carries {len(experts)}")
            if layer in by_decode[d]:
                raise TraceError(f"{path}:{lineno}: decode {d} repeats layer {layer}")
            by_decode[d][layer] = experts
            n_used_seen.add(n_used)

    if not by_decode:
        raise TraceError(f"{path}: no routing records")

    layers = sorted({l for d in by_decode.values() for l in d})
    out: list[TokenRouting] = []
    for d in sorted(by_decode):
        if sorted(by_decode[d]) != layers:
            missing = sorted(set(layers) - set(by_decode[d]))
            raise TraceError(f"{path}: decode {d} is missing layers {missing}")
        width = max(len(rows) for rows in by_decode[d].values())
        is_prompt = (d == 0)
        if (is_prompt and not prompt) or (not is_prompt and not generated):
            continue
        for i in range(width):
            routing: dict[int, tuple[int, ...]] = {}
            for layer in layers:
                rows = by_decode[d][layer]
                # A short layer covers the decode's last tokens, so its row
                # for token i sits at i - (width - len(rows)).
                offset = width - len(rows)
                if offset < 0:
                    raise TraceError(f"{path}: decode {d} layer {layer} has more rows "
                                     f"({len(rows)}) than the decode is wide ({width})")
                j = i - offset
                if 0 <= j < len(rows):
                    routing[layer] = tuple(rows[j])
            if routing:
                out.append(TokenRouting(decode=d, index=i, prompt=is_prompt, routing=routing))
    return out


def describe(path: str | Path) -> TraceInfo:
    """What a trace contains, without assembling it twice."""
    rows = read_trace(path)
    info = TraceInfo(path=str(path))
    info.decodes = len({r.decode for r in rows})
    info.prompt_tokens = sum(1 for r in rows if r.prompt)
    info.generated_tokens = sum(1 for r in rows if not r.prompt)
    info.layers = tuple(sorted({l for r in rows for l in r.routing}))
    info.n_used = max(len(e) for r in rows for e in r.routing.values())
    info.partial_tokens = sum(1 for r in rows if len(r.routing) < len(info.layers))
    info.experts_seen = len({e for r in rows for es in r.routing.values() for e in es})
    return info


def routings(rows: Sequence[TokenRouting]) -> Iterator[dict[int, Sequence[int]]]:
    """The mapping sequence the predictors and the cache benchmark consume."""
    for r in rows:
        yield r.as_mapping()

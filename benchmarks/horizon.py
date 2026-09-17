#!/usr/bin/env python3
"""How much of the model does a window of tokens actually need?

This is the measurement the whole project turns on. If a token's experts are
a small slice of the file, tiering can work; if a handful of tokens touch
most of it, there is nothing to tier and the only honest thing is to say so.

    python benchmarks/horizon.py TRACE.jsonl [--model MODEL.gguf]

For each window width W it reports the largest set of distinct experts any W
consecutive tokens needed — the largest, not the average, because a residency
budget has to survive the worst window it meets, not the typical one.

With a model given, the sizes are real bytes and the always-resident floor
(attention, shared expert, router, embeddings) is added, so the figures are
what a runtime would actually have to hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.trace import describe, read_trace  # noqa: E402

GB = 1024 ** 3


def widest(sets, w: int, cost=None, samples: int = 200):
    """The widest window of W consecutive tokens: its expert count and bytes.

    Widest by bytes when a cost function is given, because experts are not
    all the same size — taking the first MoE layer's size for every layer
    overstated this model's expert total by 8%.
    """
    if w > len(sets):
        return 0, 0
    step = max(1, (len(sets) - w) // samples or 1)
    best_n, best_bytes = 0, 0
    for i in range(0, len(sets) - w + 1, step):
        u = set().union(*sets[i:i + w])
        b = sum(cost(k) for k in u) if cost else 0
        if (b, len(u)) > (best_bytes, best_n):
            best_n, best_bytes = len(u), b
    return best_n, best_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--prompt-tokens", action="store_true")
    a = ap.parse_args()

    info = describe(a.trace)
    rows = read_trace(a.trace, prompt=a.prompt_tokens)
    sets = [{(l, e) for l, es in r.routing.items() for e in es} for r in rows]
    slots = len(info.layers) * (max(e for s in sets for _, e in s) + 1)

    cost = floor_gb = routed_gb = None
    if a.model:
        from tierinfer.gguf import read_gguf
        from tierinfer.index import ModelIndex
        ix = ModelIndex(read_gguf(a.model))
        floor_gb = ix.always_resident_nbytes() / GB
        routed_gb = ix.routed_nbytes() / GB
        sizes: dict[tuple[int, int], int] = {}

        def cost(key):
            if key not in sizes:
                try:
                    sizes[key] = ix.expert(*key).nbytes
                except Exception:
                    sizes[key] = 0
            return sizes[key]

    print(f"{a.trace.name}: {len(sets)} tokens, {len(info.layers)} MoE layers, "
          f"{info.n_used} routed per layer, {slots} expert slots")
    if a.model:
        print(f"{a.model.name}: {floor_gb:.2f} GB always resident + "
              f"{routed_gb:.2f} GB of experts")
    print()

    head = f"{'window':>8}{'experts':>10}{'of model':>10}"
    if a.model:
        head += f"{'experts GB':>13}{'+floor GB':>12}{'of file':>10}"
    print(head)

    widths = [w for w in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if w <= len(sets)]
    if len(sets) not in widths:
        widths.append(len(sets))
    for w in widths:
        n, nbytes = widest(sets, w, cost)
        line = f"{w:>8}{n:>10}{n / slots:>10.1%}"
        if a.model:
            eg = nbytes / GB
            line += (f"{eg:>13.2f}{eg + floor_gb:>12.2f}"
                     f"{(eg + floor_gb) / (floor_gb + routed_gb):>10.1%}")
        print(line)

    overlap = [len(sets[i] & sets[i + 1]) for i in range(len(sets) - 1)]
    if overlap:
        mean = sum(overlap) / len(overlap)
        per_token = len(info.layers) * info.n_used
        print(f"\nconsecutive tokens share {mean:.0f} of {per_token} experts "
              f"({mean / per_token:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

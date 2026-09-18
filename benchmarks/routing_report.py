#!/usr/bin/env python3
"""What a captured routing trace says about the model — and how predictable it is.

One script for addendum §13 and §14, so the numbers for both come from the
same rows. For each trace: how many experts a token actually touches (the
measured working set, against the layout's), how skewed activation is per
layer, how much of the file a window of W tokens needs (the horizon), how
much consecutive tokens share, and then recall@k for every predictor in
`tierinfer.predict` on generated tokens only — prompt tokens arrive in one
batch and nothing about them is being predicted ahead.

    python benchmarks/routing_report.py SHARD.gguf TRACE.jsonl [TRACE2.jsonl ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.index import load  # noqa: E402
from tierinfer.predict import (AdaptiveBlend, Frequency, Persistence, Transition,  # noqa: E402
                               evaluate)
from tierinfer.trace import describe, read_trace, routings  # noqa: E402

GB = 1024 ** 3


def window_bytes(sets, w, per_expert, samples=200, seed=1):
    """Median bytes touched by W consecutive tokens, sampled over the trace."""
    if len(sets) < w:
        return None
    rng = random.Random(seed)
    starts = range(len(sets) - w + 1)
    picks = list(starts) if len(starts) <= samples else rng.sample(list(starts), samples)
    sizes = []
    for s in picks:
        union = set()
        for t in sets[s:s + w]:
            union |= t
        sizes.append(len(union) * per_expert)
    return statistics.median(sizes)


def report(ix, trace_path, k_values=(8, 16, 32), warmup=50):
    info = describe(trace_path)
    rows = read_trace(trace_path, prompt=False)
    per_expert = ix.expert_nbytes()
    floor = ix.always_resident_nbytes()
    n_layers = len(ix.moe_layers)
    total_experts = n_layers * ix.expert_count

    sets = [frozenset((l, e) for l, es in r.routing.items() for e in es) for r in rows]
    per_token = [len(s) for s in sets]

    # activation skew per layer: share of a layer's activations taken by its top 10 % experts
    counts = defaultdict(Counter)
    for r in rows:
        for l, es in r.routing.items():
            counts[l].update(es)
    top10 = []
    never = 0
    for l, c in counts.items():
        tot = sum(c.values())
        top = sum(n for _, n in c.most_common(max(1, ix.expert_count // 10)))
        top10.append(top / tot if tot else 0)
        never += ix.expert_count - len(c)

    # neighbour overlap
    overlaps = [len(a & b) / len(a) for a, b in zip(sets, sets[1:]) if a]

    out = {
        "trace": str(trace_path), "tokens_generated": len(rows), "tokens_prompt": info.prompt_tokens,
        "layers": n_layers, "experts_per_layer": ix.expert_count, "used_per_layer": ix.expert_used_count,
        "experts_per_token_layout": n_layers * ix.expert_used_count,
        "experts_per_token_measured_median": statistics.median(per_token) if per_token else 0,
        "working_set_gb_layout": (floor + n_layers * ix.expert_used_count * per_expert) / GB,
        "working_set_gb_measured": (floor + statistics.median(per_token) * per_expert) / GB if per_token else 0,
        "distinct_experts_seen": len({k for s in sets for k in s}),
        "distinct_share": len({k for s in sets for k in s}) / total_experts,
        "experts_never_routed": never,
        "top10pct_share_median": statistics.median(top10) if top10 else 0,
        "top10pct_share_uniform_would_be": 0.10,
        "neighbour_overlap_median": statistics.median(overlaps) if overlaps else 0,
        "horizon": {},
        "predictors": {},
    }
    for w in (1, 2, 4, 8, 16, 32, 64, 128):
        b = window_bytes(sets, w, per_expert)
        if b is not None:
            out["horizon"][w] = {"gb": (floor + b) / GB, "share_of_file": (floor + b) / ix.gguf.nbytes_on_disk}

    trace = list(routings(rows))
    for k in k_values:
        res = {}
        for name, mk in (("frequency", Frequency), ("persistence", Persistence), ("transition", Transition),
                         ("adaptive", lambda: AdaptiveBlend([Frequency(), Persistence(), Transition()], k=k))):
            sc = evaluate(mk(), trace, k=k, warmup=warmup)
            res[name] = {"recall": sc.recall, "wasted": sc.wasted, "tokens": sc.tokens,
                         "hits": sc.hits, "used": sc.used, "predicted": sc.predicted}
        out["predictors"][k] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("traces", type=Path, nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--warmup", type=int, default=50)
    a = ap.parse_args()
    ix = load(a.model)
    reports = [report(ix, t, warmup=a.warmup) for t in a.traces]
    for r in reports:
        print(f"\n== {Path(r['trace']).name}: {r['tokens_generated']} generated tokens "
              f"(+{r['tokens_prompt']} prompt), {r['layers']} MoE layers x {r['experts_per_layer']} experts")
        print(f"  experts per token: layout {r['experts_per_token_layout']}, measured median "
              f"{r['experts_per_token_measured_median']:.0f}  ->  working set "
              f"{r['working_set_gb_measured']:.2f} GB measured vs {r['working_set_gb_layout']:.2f} GB layout")
        print(f"  distinct experts seen: {r['distinct_experts_seen']} of {r['layers'] * r['experts_per_layer']} "
              f"({r['distinct_share']:.1%}); never routed: {r['experts_never_routed']}")
        print(f"  activation skew: top 10% of a layer's experts take {r['top10pct_share_median']:.1%} of its "
              f"activations (uniform: 10%); neighbour tokens share {r['neighbour_overlap_median']:.1%} of experts")
        print("  horizon (median GB a window of W tokens needs, share of file):")
        print("    " + "  ".join(f"W={w}: {h['gb']:.1f} GB ({h['share_of_file']:.1%})" for w, h in r["horizon"].items()))
        for k, res in r["predictors"].items():
            print(f"  recall@{k}: " + "  ".join(f"{n} {v['recall']:.1%}" for n, v in res.items())
                  + f"   (waste@{k}: adaptive {res['adaptive']['wasted']:.1%})")
    if a.json:
        a.json.write_text(json.dumps(reports, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

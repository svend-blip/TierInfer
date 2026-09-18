#!/usr/bin/env python3
"""Does a trained prerouter beat the counting predictors on real routing?

Trains `tierinfer.prerouter.Prerouter` on one trace (or the first half of
one) and scores recall@k on another (or the second half), against
`Transition`, `Persistence` and the adaptive blend given the same training
tokens. Both directions across prompt classes, so a model that only knows
prose is tested on code and the other way round.

    python benchmarks/prerouter_eval.py SHARD.gguf traces/a.jsonl traces/b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.index import load  # noqa: E402
from tierinfer.predict import AdaptiveBlend, Frequency, Persistence, Transition, evaluate  # noqa: E402
from tierinfer.prerouter import Prerouter  # noqa: E402
from tierinfer.trace import read_trace, routings  # noqa: E402


def counting(name):
    return {"transition": Transition, "persistence": Persistence, "frequency": Frequency,
            "adaptive": lambda: AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16)}[name]()


def run(ix, train, test, k, epochs, label):
    rows = {}
    t0 = time.perf_counter()
    pr = Prerouter(ix.moe_layers, ix.expert_count, lr=0.05)
    losses = pr.train(train, epochs=epochs)
    train_s = time.perf_counter() - t0
    pr.online = False
    rows["prerouter (frozen)"] = evaluate(pr, test, k=k).recall
    pr2 = Prerouter.load(_save(pr), expect_layers=ix.moe_layers, expect_experts=ix.expert_count)
    pr2.online = True
    rows["prerouter (online)"] = evaluate(pr2, test, k=k).recall
    for name in ("transition", "persistence", "frequency", "adaptive"):
        p = counting(name)
        for r in train:
            p.observe(r)
        rows[name] = evaluate(p, test, k=k).recall
    return {"label": label, "k": k, "train_tokens": len(train), "test_tokens": len(test),
            "epochs": epochs, "train_seconds": train_s, "loss_first": losses[0], "loss_last": losses[-1],
            "recall": rows}


def _save(pr):
    import tempfile
    p = Path(tempfile.mkdtemp()) / "pr.npz"
    pr.save(p)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("traces", type=Path, nargs="+")
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    ix = load(a.model)
    traces = {t.stem: list(routings(read_trace(t, prompt=False))) for t in a.traces}
    out = []
    for k in a.k:
        for name, rows in traces.items():
            half = len(rows) // 2
            out.append(run(ix, rows[:half], rows[half:], k, a.epochs, f"{name}: first half -> second half"))
        names = list(traces)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    out.append(run(ix, traces[names[i]], traces[names[j]], k, a.epochs,
                                   f"{names[i]} -> {names[j]}"))
    print("| k | train -> test | prerouter frozen | prerouter online | transition | persistence | frequency | adaptive | train s |")
    print("|--:|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in out:
        R = r["recall"]
        print(f"| {r['k']} | {r['label']} | {R['prerouter (frozen)']:.1%} | {R['prerouter (online)']:.1%} | "
              f"{R['transition']:.1%} | {R['persistence']:.1%} | {R['frequency']:.1%} | {R['adaptive']:.1%} | {r['train_seconds']:.1f} |")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

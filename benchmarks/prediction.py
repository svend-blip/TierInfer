#!/usr/bin/env python3
"""How much of a token's routing can be known before the token runs?

Recall@k is the number prefetching lives on: of the experts a token actually
routed to, how many were in the top k of the prediction. A predicted expert
that goes unused costs bandwidth; a used expert that went unpredicted costs a
stall, and the stall is much the more expensive of the two.

Frequency is the floor. It needs no context, no state and no per-token work,
so a context-aware predictor has to beat it by enough to pay for itself.

    python benchmarks/prediction.py [--tokens 800] [--experts 128] [--layers 46]

**This scores code, not a model.** The trace comes from a generator written
alongside the predictors, so its assumptions — a slowly drifting prompt
topic, layer-to-layer correlation, a Zipf tail — are exactly the structure
the predictors look for. Real routing may have more of that structure or
less. `evaluate` takes any iterable of routings, so a captured trace from a
running model scores through the same code; until then, read this as "the
predictors do what they claim on data that has the property they assume."
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.predict import (  # noqa: E402
    AdaptiveBlend, Blend, Frequency, Persistence, Transition, evaluate,
)
from tierinfer.trace import describe, read_trace, routings  # noqa: E402


def synthetic_trace(tokens: int, layers: int, experts: int, routed: int,
                    *, drift: float = 0.08, coupling: float = 0.7,
                    seed: int = 20260917):
    """A trace with the three structures MoE routing is reported to have.

    - a prompt-level topic that drifts slowly, so routing is autocorrelated
    - layer-to-layer coupling, so an earlier layer's choice informs a later one
    - a skewed marginal, so some experts are simply more popular

    Each is a separate knob, so a predictor that only exploits one of them
    can be seen doing exactly that.
    """
    rnd = random.Random(seed)
    popularity = sorted((rnd.paretovariate(1.1) for _ in range(experts)), reverse=True)
    topic = [rnd.randrange(experts) for _ in range(layers)]
    for _ in range(tokens):
        routing = {}
        prev: list[int] | None = None
        for layer in range(layers):
            if rnd.random() < drift:
                topic[layer] = rnd.randrange(experts)
            weights = []
            for e in range(experts):
                w = popularity[e]
                if abs(e - topic[layer]) <= 2:
                    w *= 12.0                       # the topic's neighbourhood
                if prev is not None and coupling and e in {(p * 7 + 3) % experts for p in prev}:
                    w *= 1.0 + 20.0 * coupling      # coupled to the layer below
                weights.append(w)
            chosen: set[int] = set()
            while len(chosen) < routed:
                chosen.add(rnd.choices(range(experts), weights=weights, k=1)[0])
            routing[layer] = sorted(chosen)
            prev = routing[layer]
        yield routing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--layers", type=int, default=46)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--routed", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--trace", type=Path, default=None,
                    help="a captured routing trace (tools/trace); without it, "
                         "a synthetic one is generated and the result is about "
                         "the code rather than about any model")
    ap.add_argument("--prompt-tokens", action="store_true",
                    help="score prompt tokens too; by default only generated ones, "
                         "because a prompt decode needs every expert at once and "
                         "nothing is being predicted ahead")
    a = ap.parse_args()

    if a.trace:
        info = describe(a.trace)
        rows = read_trace(a.trace, prompt=a.prompt_tokens)
        trace = list(routings(rows))
        a.experts = max(e for r in trace for es in r.values() for e in es) + 1
        print(f"{a.trace.name}: {info.tokens} tokens ({info.prompt_tokens} prompt, "
              f"{info.generated_tokens} generated), {len(info.layers)} MoE layers, "
              f"{info.n_used} routed per layer, {info.experts_seen} distinct experts seen")
        print(f"scoring {len(trace)} tokens"
              f"{'' if a.prompt_tokens else ' (generated only)'}; "
              f"{a.warmup} warmup tokens not scored\n")
    else:
        trace = list(synthetic_trace(a.tokens, a.layers, a.experts, a.routed))
        print(f"SYNTHETIC — this scores the code, not a model.")
        print(f"{a.tokens} tokens, {a.layers} layers, {a.experts} experts, "
              f"{a.routed} routed per layer; {a.warmup} warmup tokens not scored\n")

    def build():
        freq, pers, trans = Frequency(), Persistence(), Transition()
        hand = Blend([(Frequency(), 0.2), (Persistence(), 0.4), (Transition(), 0.4)])
        adaptive = AdaptiveBlend([Frequency(), Persistence(), Transition()], k=k)
        return [freq, pers, trans, hand, adaptive]

    for k in (8, 16, 32):
        print(f"k = {k}  ({k / a.experts:.0%} of the layer's experts held ready)")
        print(f"  {'predictor':<34}{'recall':>9}{'wasted':>9}{'vs freq':>10}")
        floor = None
        adaptive = None
        for p in build():
            s = evaluate(p, trace, k=k, warmup=a.warmup)
            if floor is None:
                floor = s.recall
            delta = f"{s.recall - floor:+.1%}" if p.name != "frequency" else "—"
            print(f"  {p.name:<34}{s.recall:>9.1%}{s.wasted:>9.1%}{delta:>10}")
            if p.name == "adaptive":
                adaptive = p
        if adaptive is not None:
            w = adaptive.weights()
            print("  weights it settled on: " +
                  ", ".join(f"{n} {v:.0%}" for n, v in sorted(w.items())))
        print()

    # Where the prediction is weakest is where a stall will happen.
    worst = evaluate(Transition(), trace, k=16, warmup=a.warmup)
    print("transition's three worst layers at k=16:")
    for layer, rate in worst.worst_layers(3):
        print(f"  layer {layer:>3}  recall {rate:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

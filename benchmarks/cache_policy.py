#!/usr/bin/env python3
"""Does the value policy actually beat LRU, or does it only sound better?

MoE routing is skewed: some experts are chosen far more often than others.
This replays a synthetic routing trace with that shape through both policies
at the same capacity and reports the hit rates.

The trace is synthetic, and that is stated rather than hidden: a real trace
needs a running model with instrumentation, which is Goal 6. What this can
settle now is whether the policy is worth carrying into that work.

    python benchmarks/cache_policy.py [--experts 128] [--layers 46] [--skew 1.1]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.cache import ExpertCache  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402


class LRUCache:
    """The comparison: bounded by bytes, evicting by last use alone."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.used = 0
        self.order: dict[tuple[int, int], int] = {}
        self.sizes: dict[tuple[int, int], int] = {}
        self.clock = 0
        self.hits = self.misses = 0

    def get(self, key):
        self.clock += 1
        if key in self.order:
            self.order[key] = self.clock
            self.hits += 1
            return True
        self.misses += 1
        return None

    def put(self, key, nbytes):
        self.clock += 1
        while self.used + nbytes > self.capacity_bytes and self.order:
            victim = min(self.order, key=self.order.get)
            self.used -= self.sizes.pop(victim)
            del self.order[victim]
        self.order[key] = self.clock
        self.sizes[key] = nbytes
        self.used += nbytes

    @property
    def hit_rate(self):
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


def zipf_choice(rng, n, skew, k):
    """``k`` distinct draws from a Zipf-like distribution over ``n`` experts."""
    weights = [1.0 / ((i + 1) ** skew) for i in range(n)]
    chosen: set[int] = set()
    while len(chosen) < k:
        chosen.add(rng.choices(range(n), weights=weights, k=1)[0])
    return chosen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--layers", type=int, default=46)
    p.add_argument("--used", type=int, default=8, help="experts routed per layer per token")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--skew", type=float, default=1.1)
    p.add_argument("--expert-mb", type=float, default=8.94)
    p.add_argument("--cache-gb", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trace", type=Path, default=None,
                   help="a captured routing trace (tools/trace); without it the "
                        "trace is synthetic and the result is about the code")
    p.add_argument("--prompt-tokens", action="store_true")
    p.add_argument("--w-recency", type=float, default=0.0,
                   help="weight on the recency term in the value function; 0 is "
                        "the original rate-only policy")
    p.add_argument("--w-rate", type=float, default=1.0)
    p.add_argument("--recency-decay", type=float, default=0.9)
    a = p.parse_args(argv)

    rng = random.Random(a.seed)
    size = int(a.expert_mb * 1024 * 1024)
    capacity = int(a.cache_gb * 1024 ** 3)

    # One trace, replayed through both policies, so the comparison is fair.
    if a.trace:
        from tierinfer.trace import describe, read_trace
        info = describe(a.trace)
        rows = read_trace(a.trace, prompt=a.prompt_tokens)
        trace = [[(l, e) for l, es in r.routing.items() for e in es] for r in rows]
        a.layers, a.experts = len(info.layers), info.experts_seen
        a.used, a.tokens = info.n_used, len(trace)
        source = (f"{a.trace.name}: {info.tokens} tokens "
                  f"({info.generated_tokens} generated), measured routing")
    else:
        trace = []
        for _ in range(a.tokens):
            token = []
            for layer in range(a.layers):
                token += [(layer, e) for e in zipf_choice(rng, a.experts, a.skew, a.used)]
            trace.append(token)
        source = f"SYNTHETIC, skew {a.skew} — this scores the code, not a model"

    tracker = ExpertTracker(window=128)
    value_cache = ExpertCache(capacity, tracker, w_rate=a.w_rate,
                              w_recency=a.w_recency, recency_decay=a.recency_decay)
    lru = LRUCache(capacity)

    for token in trace:
        tracker.begin_token()
        for key in token:
            if value_cache.get(key) is None:
                value_cache.put(key, None, size)
            if lru.get(key) is None:
                lru.put(key, size)
        tracker.record(token)

    total = a.layers * a.experts
    resident = capacity / size
    print(source)
    print(f"value policy: w_rate {a.w_rate:g}, w_recency {a.w_recency:g}, "
          f"decay {a.recency_decay:g}")
    print(f"{a.experts} experts x {a.layers} layers = {total} slots, "
          f"{a.used} routed per layer per token")
    print(f"expert {a.expert_mb:.2f} MB, cache {a.cache_gb:.1f} GB = "
          f"{resident:.0f} experts resident ({resident / total:.0%} of the model)")
    print(f"{a.tokens} tokens, {len(trace[0])} activations per token\n")
    print(f"{'policy':<10} {'hit rate':>9} {'hits':>10} {'misses':>10} {'evictions':>10}")
    print(f"{'value':<10} {value_cache.stats.hit_rate:>8.1%} {value_cache.stats.hits:>10} "
          f"{value_cache.stats.misses:>10} {value_cache.stats.evictions:>10}")
    print(f"{'LRU':<10} {lru.hit_rate:>8.1%} {lru.hits:>10} {lru.misses:>10} "
          f"{'—':>10}")
    delta = value_cache.stats.hit_rate - lru.hit_rate
    print(f"\ndifference: {delta:+.1%} in hit rate")
    if abs(delta) < 0.005:
        print("No meaningful difference at this capacity and skew.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

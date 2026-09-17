#!/usr/bin/env python3
"""Does one adaptive policy beat fixed ones, on real routing?

Four arms over the same trace and the same budgets, differing only in how
they decide what to hold and what to fetch ahead:

    nvme-only     nothing is retained; every expert is read when needed
    ram-only      a RAM cache, no VRAM tier, no speculation
    fixed-N       both tiers, prefetch depth pinned at N
    adaptive      both tiers, depth moved by measured stalls against waste

Costs are the measured per-tier transfer times (`TierCosts`), applied to the
real sequence of expert activations. This is a cost model over measured
constants, not an inference run: it answers "how much transfer time does this
policy incur", which is the part a policy controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from tierinfer.cache import ExpertCache  # noqa: E402
from tierinfer.gguf import read_gguf  # noqa: E402
from tierinfer.index import ModelIndex  # noqa: E402
from tierinfer.policy import PolicyStats, Tier, TierCosts, TierPolicy  # noqa: E402
from tierinfer.predict import AdaptiveBlend, Frequency, Persistence, Transition  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402
from tierinfer.trace import read_trace  # noqa: E402
from tierinfer.vram import GB, MB, VramBudget  # noqa: E402

WARM_TOKEN_SECONDS = 1.0 / 5.90


class SimTier:
    """A bounded set of keys with LRU eviction — the tiers' shape, not their bytes."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self._order: dict = {}

    def __contains__(self, key) -> bool:
        if key in self._order:
            self._order[key] = self._order.pop(key)   # move to the back
            return True
        return False

    def __len__(self) -> int:
        return len(self._order)

    def admit(self, key) -> None:
        if self.slots <= 0:
            return
        self._order.pop(key, None)
        self._order[key] = None
        while len(self._order) > self.slots:
            self._order.pop(next(iter(self._order)))


def run(tokens, vram_slots, ram_slots, *, depth, adaptive, costs):
    tracker = ExpertTracker(window=128)
    predictor = AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16)
    policy = TierPolicy(SimTier(vram_slots), SimTier(ram_slots), predictor, tracker,
                        costs, depth=depth,
                        min_depth=0 if adaptive else depth,
                        max_depth=32 if adaptive else depth)
    for routing in tokens:
        sofar: dict = {}
        for layer in sorted(routing):
            policy.before_layer(layer, sofar)
            policy.on_routing(layer, routing[layer])
            sofar[layer] = routing[layer]
        tracker.record([(l, e) for l, es in routing.items() for e in es])
        predictor.observe(routing)
        policy.end_token()
    return policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--ram-gb", type=float, default=64.0)
    ap.add_argument("--vram-gb", type=float, default=0.0,
                    help="0 means derive it from the model and the context")
    a = ap.parse_args()

    ix = ModelIndex(read_gguf(a.model))
    expert = ix.expert_nbytes()
    floor = ix.always_resident_nbytes()
    rows = read_trace(a.trace, prompt=False)
    tokens = [r.as_mapping() for r in rows]

    if a.vram_gb:
        vram_slots = int(a.vram_gb * GB) // expert
    else:
        b = VramBudget.from_model(ix.gguf.metadata, total_bytes=31.36 * GB,
                                  context_length=a.context, layers=ix.block_count,
                                  reserve_bytes=GB)
        vram_slots = b.experts(expert, floor)
    ram_slots = int(a.ram_gb * GB) // expert
    costs = TierCosts()

    per_token = sum(len(v) for v in tokens[0].values())
    print(f"{a.trace.name}: {len(tokens)} tokens, {per_token} activations each")
    print(f"VRAM {vram_slots} experts ({vram_slots * expert / GB:.1f} GB), "
          f"RAM {ram_slots} ({ram_slots * expert / GB:.1f} GB), "
          f"model {(floor + ix.routed_nbytes()) / GB:.1f} GB")
    print(f"costs: VRAM {costs.vram_hit * 1000:.2f} ms, RAM {costs.ram_to_vram * 1000:.2f} ms, "
          f"NVMe {costs.cost_from(Tier.NVME) * 1000:.2f} ms "
          f"({costs.tier_ratio:.0f}x)\n")

    print(f"{'policy':<14}{'resident':>10}{'VRAM hits':>11}{'stalls':>9}"
          f"{'prefetch':>10}{'ms/token':>10}{'of warm':>9}{'depth':>8}")

    arms = [("nvme-only", 0, 0, 0, False),
            ("ram-only", 0, ram_slots, 0, False),
            ("fixed-0", vram_slots, ram_slots, 0, False),
            ("fixed-8", vram_slots, ram_slots, 8, False),
            ("fixed-16", vram_slots, ram_slots, 16, False),
            ("adaptive", vram_slots, ram_slots, 8, True)]

    best = None
    for label, vs, rs, depth, adapt in arms:
        p = run(tokens, vs, rs, depth=depth, adaptive=adapt, costs=costs)
        s = p.stats
        ms = s.seconds_per_token * 1000
        acc = f"{s.prefetch_accuracy:.0%}" if s.prefetched else "—"
        print(f"{label:<14}{s.resident_rate:>10.1%}{s.vram_hit_rate:>11.1%}"
              f"{s.stalls / max(s.tokens, 1):>9.0f}{acc:>10}{ms:>10.1f}"
              f"{ms / 1000 / WARM_TOKEN_SECONDS:>8.0%}{p.depth:>8}")
        if best is None or ms < best[1]:
            best = (label, ms)
        if adapt:
            moves = p.depth_history
            print(f"{'':<14}depth moved {s.depth_changes} times: "
                  + " -> ".join(f"{d}@{t}" for t, d in moves[:6])
                  + (" ..." if len(moves) > 6 else ""))

    print(f"\nbest: {best[0]} at {best[1]:.1f} ms/token")
    print("a warm unconstrained token is 169 ms (BASELINE.md); these are the "
          "transfer costs a policy controls, applied to real routing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One policy across VRAM, RAM and NVMe, adapting while it runs — **as a simulation**.

**What this module is, stated first because the audit found it stated last.**
`TierPolicy` is a cost model. `on_routing` returns seconds computed from the
constants in `TierCosts`; it moves no bytes and observes no transfer. Its
tiers are anything with ``key in tier`` and ``tier.admit(key)`` — which the
real `ExpertCache` (no ``admit``) and `VramResidency` (``admit`` takes a host
pointer and a size) are not, so it has only ever run over `SimTier`
(`benchmarks/policy.py`). `PolicyStats.seconds` and ``prefetch_seconds`` are
modelled and are exported under the ``sim.`` telemetry namespace, not beside
observed counters. `benchmarks/POLICY.md` says the same in its last section.
The runtime policy over real tiers is the loader's job (audit, item 12).

Everything else in this project decides one thing well. The tracker knows
what has been used, the predictor ranks what is coming, the caches hold what
fits, the streamer moves bytes. None of them knows what an expert is *worth*,
because worth depends on where it currently is and what it would cost to get
it from there — and that is the only question a tier policy answers.

The costs are measured, not assumed, and they are nine times apart:

    VRAM hit                    0 ms    it is already there
    RAM → VRAM (pinned)      0.36 ms    27.6 GB/s, benchmarks/VRAM.md
    NVMe → RAM (8 workers)   3.30 ms    3.0 GB/s, benchmarks/STREAMING.md

That ratio is the whole shape of the policy. An expert one tier down costs a
tenth of what the same expert costs two tiers down, so the interesting
decision is almost never "VRAM or RAM" — it is "in RAM at all, or not".

**What adapts, and against what.** One parameter is free: how many predicted
experts to fetch ahead of a layer. Too few and the token waits; too many and
bandwidth goes on experts nobody asked for. Both costs are measurable while
running, so the depth is not configured — it is moved by what the last few
hundred tokens actually cost, the same mechanism `AdaptiveBlend` uses for
predictor weights and for the same reason: a fixed choice would be this
host's number baked in as if it were a law.

**What does not adapt.** Correctness. A routed expert is always delivered: if
prediction missed it and no tier holds it, the policy issues an exact read
and counts a stall. No amount of adaptation is allowed to turn a miss into a
wrong answer — SCOPE goal 15, one level up from where `prefetch.py` states
the same rule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .predict import Predictor
from .tracker import ExpertKey, ExpertTracker


class Tier(Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"


@dataclass(frozen=True)
class TierCosts:
    """Seconds to make an expert usable, by where it currently is.

    The defaults are this host's measurements, named so they can be replaced
    rather than trusted: a different card, bus or drive moves all three.
    """

    vram_hit: float = 0.0
    ram_to_vram: float = 0.00036      # pinned, 27.6 GB/s over 9.97 MB
    nvme_to_ram: float = 0.00330      # 8 workers, 3.0 GB/s over 9.97 MB

    def cost_from(self, tier: Tier) -> float:
        if tier is Tier.VRAM:
            return self.vram_hit
        if tier is Tier.RAM:
            return self.ram_to_vram
        return self.nvme_to_ram + self.ram_to_vram

    @property
    def tier_ratio(self) -> float:
        """How much worse the bottom tier is than the middle one."""
        return self.cost_from(Tier.NVME) / self.cost_from(Tier.RAM)


@dataclass
class PolicyStats:
    tokens: int = 0
    lookups: int = 0
    vram_hits: int = 0
    ram_hits: int = 0
    nvme_reads: int = 0
    prefetched: int = 0
    prefetch_used: int = 0
    stalls: int = 0
    seconds: float = 0.0
    prefetch_seconds: float = 0.0
    depth_changes: int = 0

    @property
    def vram_hit_rate(self) -> float:
        return self.vram_hits / self.lookups if self.lookups else 0.0

    @property
    def resident_rate(self) -> float:
        """Share served without touching NVMe."""
        return (self.vram_hits + self.ram_hits) / self.lookups if self.lookups else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        return self.prefetch_used / self.prefetched if self.prefetched else 0.0

    @property
    def seconds_per_token(self) -> float:
        return self.seconds / self.tokens if self.tokens else 0.0

    @property
    def wasted_prefetch_seconds(self) -> float:
        return self.prefetch_seconds * (1.0 - self.prefetch_accuracy)


class TierPolicy:
    """Places experts across three tiers, and moves its own dial while it runs.

    The tiers are supplied as membership-and-admit objects rather than
    concrete classes, so this is testable without a GPU and reusable over a
    real `VramResidency` and `ExpertCache`. What it requires of them is small:
    ``key in tier``, ``tier.admit(key)`` and ``tier.evict_if_needed()``.
    """

    def __init__(self, vram, ram, predictor: Predictor, tracker: ExpertTracker,
                 costs: TierCosts | None = None, *, depth: int = 8,
                 min_depth: int = 0, max_depth: int = 32,
                 window_tokens: int = 32) -> None:
        if not 0 <= min_depth <= depth <= max_depth:
            raise ValueError("depth must sit between min_depth and max_depth")
        self.vram = vram
        self.ram = ram
        self.predictor = predictor
        self.tracker = tracker
        self.costs = costs or TierCosts()
        self.depth = depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.stats = PolicyStats()
        #: Per-token (stall_seconds, wasted_prefetch_seconds) over the window
        #: the depth is adapted against.
        self.window: deque[tuple[float, float]] = deque(maxlen=window_tokens)
        self.depth_history: list[tuple[int, int]] = [(0, depth)]
        self._pending: dict[ExpertKey, float] = {}
        self._token_stall = 0.0
        self._token_waste = 0.0

    # -- where an expert is ---------------------------------------------

    def locate(self, key: ExpertKey) -> Tier:
        if key in self.vram:
            return Tier.VRAM
        if key in self.ram:
            return Tier.RAM
        return Tier.NVME

    def value_of_holding(self, key: ExpertKey, tier: Tier) -> float:
        """Expected seconds saved by having this expert one tier up.

        Probability comes from the tracker's measured activation rate rather
        than from the predictor: this is a question about *retention*, and on
        real routing prediction confidence moved retention by a tenth of a
        point (`benchmarks/REAL-ROUTING.md`). Prediction earns its place in
        the fetch decision below, not here.
        """
        p = self.tracker.activation_rate(key) if self.tracker else 0.0
        if tier is Tier.VRAM:
            return 0.0
        if tier is Tier.RAM:
            return p * (self.costs.ram_to_vram - self.costs.vram_hit)
        return p * (self.costs.cost_from(Tier.NVME) - self.costs.ram_to_vram)

    # -- speculation ----------------------------------------------------

    def before_layer(self, layer: int, sofar: Mapping[int, Sequence[int]]) -> list[ExpertKey]:
        """Fetch what this layer is likely to want, as deep as the dial says."""
        if self.depth <= 0:
            return []
        issued: list[ExpertKey] = []
        prediction = self.predictor.predict(layer, dict(sofar))
        for expert in prediction.top(self.depth):
            key = (layer, expert)
            if key in self._pending:
                continue
            tier = self.locate(key)
            if tier is Tier.VRAM:
                continue
            cost = self.costs.cost_from(tier)
            self._admit(key, tier)
            self._pending[key] = cost
            self.stats.prefetched += 1
            self.stats.prefetch_seconds += cost
            self._token_waste += cost
            issued.append(key)
        return issued

    # -- what actually happened -----------------------------------------

    def on_routing(self, layer: int, experts: Sequence[int]) -> float:
        """Resolve a layer's real routing. Returns the seconds it cost."""
        spent = 0.0
        for expert in experts:
            key = (layer, expert)
            self.stats.lookups += 1
            prefetched = self._pending.pop(key, None)
            if prefetched is not None:
                # Paid for already, as speculation. It was not wasted.
                self.stats.prefetch_used += 1
                self._token_waste -= prefetched
                self.stats.vram_hits += 1
                continue
            tier = self.locate(key)
            cost = self.costs.cost_from(tier)
            if tier is Tier.VRAM:
                self.stats.vram_hits += 1
            elif tier is Tier.RAM:
                self.stats.ram_hits += 1
                self._admit(key, tier)
            else:
                # Nothing anticipated it and no tier holds it. Read it
                # exactly, and call it what it is.
                self.stats.nvme_reads += 1
                self.stats.stalls += 1
                self._admit(key, tier)
            spent += cost
            self._token_stall += cost
        self.stats.seconds += spent
        return spent

    def _admit(self, key: ExpertKey, from_tier: Tier) -> None:
        if from_tier is Tier.NVME:
            self.ram.admit(key)
        if key not in self.vram:
            self.vram.admit(key)

    # -- the dial -------------------------------------------------------

    def end_token(self) -> None:
        """Close the token and move the prefetch depth if the evidence says to.

        The rule is a comparison of two measured quantities over the window,
        not a schedule: when stalls dominate waste there is room to speculate
        further, and when waste dominates stalls there is not. A dead band
        keeps it from oscillating on noise.
        """
        self.stats.tokens += 1
        self.stats.seconds += max(0.0, self._token_waste)
        self.window.append((self._token_stall, max(0.0, self._token_waste)))
        self._token_stall = 0.0
        self._token_waste = 0.0
        self._pending.clear()
        if self.tracker:
            self.tracker.begin_token()
        if len(self.window) < self.window.maxlen:
            return
        stalls = sum(s for s, _ in self.window)
        waste = sum(w for _, w in self.window)
        before = self.depth
        if stalls > waste * 1.5 and self.depth < self.max_depth:
            self.depth += 1
        elif waste > stalls * 1.5 and self.depth > self.min_depth:
            self.depth -= 1
        if self.depth != before:
            self.stats.depth_changes += 1
            self.depth_history.append((self.stats.tokens, self.depth))
            self.window.clear()      # judge the new depth on its own evidence

"""Tests for the tier policy.

Two things matter more than the arithmetic. A routed expert is always
delivered, whatever the policy guessed — and the dial moves on evidence
rather than on a schedule, in both directions.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.policy import PolicyStats, Tier, TierCosts, TierPolicy  # noqa: E402
from tierinfer.predict import Frequency, Prediction, Predictor  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402


class Bounded:
    def __init__(self, slots):
        self.slots = slots
        self._order = {}

    def __contains__(self, key):
        if key in self._order:
            self._order[key] = self._order.pop(key)
            return True
        return False

    def __len__(self):
        return len(self._order)

    def admit(self, key):
        if self.slots <= 0:
            return
        self._order.pop(key, None)
        self._order[key] = None
        while len(self._order) > self.slots:
            self._order.pop(next(iter(self._order)))


class Fixed(Predictor):
    name = "fixed"

    def __init__(self, experts):
        self.experts = list(experts)

    def observe(self, routing):
        pass

    def predict(self, layer, sofar=None):
        n = len(self.experts) or 1
        return Prediction(layer, tuple(self.experts), tuple(1 / n for _ in self.experts))

    def score(self, layer, sofar):
        return {e: 1.0 for e in self.experts}


def policy(vram=4, ram=8, depth=0, **kw):
    t = ExpertTracker(window=16)
    return TierPolicy(Bounded(vram), Bounded(ram), Fixed(kw.pop("predict", [0, 1])),
                      t, depth=depth, **kw)


# -- costs --------------------------------------------------------------


def test_the_tiers_are_an_order_of_magnitude_apart():
    c = TierCosts()
    assert c.cost_from(Tier.VRAM) == 0.0
    assert c.cost_from(Tier.RAM) < c.cost_from(Tier.NVME)
    assert c.tier_ratio > 5, "the measured ratio is what shapes the policy"


def test_reaching_an_expert_from_nvme_pays_both_hops():
    c = TierCosts(ram_to_vram=0.001, nvme_to_ram=0.010)
    assert c.cost_from(Tier.NVME) == pytest.approx(0.011)


# -- placement ----------------------------------------------------------


def test_an_expert_is_located_in_the_highest_tier_holding_it():
    p = policy()
    assert p.locate((0, 5)) is Tier.NVME
    p.ram.admit((0, 5))
    assert p.locate((0, 5)) is Tier.RAM
    p.vram.admit((0, 5))
    assert p.locate((0, 5)) is Tier.VRAM


def test_holding_something_already_in_vram_is_worth_nothing():
    p = policy()
    assert p.value_of_holding((0, 1), Tier.VRAM) == 0.0


def test_the_deeper_the_tier_the_more_holding_is_worth():
    p = policy()
    for _ in range(4):
        p.tracker.begin_token()
        p.tracker.record([(0, 1)])
    assert p.value_of_holding((0, 1), Tier.NVME) > p.value_of_holding((0, 1), Tier.RAM) > 0


def test_an_expert_never_seen_is_worth_nothing_to_hold():
    assert policy().value_of_holding((9, 9), Tier.NVME) == 0.0


# -- the guarantee ------------------------------------------------------


def test_an_unpredicted_expert_is_still_served_and_counted_a_stall():
    p = policy(depth=2, predict=[0])
    p.before_layer(0, {})
    p.on_routing(0, [7])              # 7 was never predicted and is nowhere
    assert p.stats.stalls == 1
    assert p.stats.nvme_reads == 1
    assert (0, 7) in p.ram, "the fallback must leave it resident"


def test_every_routed_expert_is_accounted_for_however_wrong_the_guess():
    p = policy(depth=4, predict=[9])
    p.before_layer(0, {})
    p.on_routing(0, [1, 2, 3])
    assert p.stats.lookups == 3


def test_a_prefetch_that_was_right_is_not_paid_for_twice():
    p = policy(depth=2, predict=[1])
    p.before_layer(0, {})
    before = p.stats.seconds
    p.on_routing(0, [1])
    assert p.stats.prefetch_used == 1
    assert p.stats.seconds == before, "a used prefetch was already paid for"


def test_a_hit_in_ram_costs_less_than_a_read_from_nvme():
    a = policy(depth=0)
    a.ram.admit((0, 1))
    ram_cost = a.on_routing(0, [1])
    b = policy(depth=0)
    nvme_cost = b.on_routing(0, [1])
    assert ram_cost < nvme_cost


def test_a_hit_in_vram_costs_nothing():
    p = policy(depth=0)
    p.vram.admit((0, 1))
    assert p.on_routing(0, [1]) == 0.0
    assert p.stats.vram_hits == 1


# -- the dial -----------------------------------------------------------


def test_the_depth_rises_when_stalls_dominate():
    p = policy(vram=0, ram=0, depth=1, min_depth=0, max_depth=8, window_tokens=4,
               predict=[])
    for _ in range(20):
        p.on_routing(0, [1, 2, 3])     # always from NVMe: pure stalls, no waste
        p.end_token()
    assert p.depth > 1
    assert p.stats.depth_changes > 0


def test_the_depth_falls_when_waste_dominates():
    p = policy(vram=64, ram=64, depth=8, min_depth=0, max_depth=8, window_tokens=4,
               predict=list(range(8)))
    for _ in range(20):
        p.before_layer(0, {})
        p.on_routing(0, [99])          # nothing predicted is ever used
        p.end_token()
    assert p.depth < 8


def test_the_depth_stays_inside_its_bounds():
    p = policy(vram=0, ram=0, depth=2, min_depth=2, max_depth=2, window_tokens=2,
               predict=[])
    for _ in range(20):
        p.on_routing(0, [1])
        p.end_token()
    assert p.depth == 2
    assert p.stats.depth_changes == 0


def test_a_change_restarts_the_evidence_rather_than_reusing_it():
    """A new depth judged on the old depth's window would chase itself."""
    p = policy(vram=0, ram=0, depth=1, min_depth=0, max_depth=8, window_tokens=4,
               predict=[])
    for _ in range(5):
        p.on_routing(0, [1, 2])
        p.end_token()
    assert p.stats.depth_changes >= 1
    assert len(p.window) < p.window.maxlen


def test_impossible_bounds_are_refused():
    with pytest.raises(ValueError):
        policy(depth=4, min_depth=8, max_depth=16)
    with pytest.raises(ValueError):
        policy(depth=20, min_depth=0, max_depth=8)


def test_depth_zero_speculates_not_at_all():
    p = policy(depth=0, predict=[0, 1, 2])
    assert p.before_layer(0, {}) == []
    assert p.stats.prefetched == 0


# -- reporting ----------------------------------------------------------


def test_an_untouched_policy_reports_zeros_rather_than_dividing_by_zero():
    s = PolicyStats()
    for v in (s.vram_hit_rate, s.resident_rate, s.prefetch_accuracy,
              s.seconds_per_token, s.wasted_prefetch_seconds):
        assert v == 0.0


def test_waste_is_the_share_of_prefetching_that_went_unused():
    s = PolicyStats(prefetched=10, prefetch_used=4, prefetch_seconds=1.0)
    assert s.prefetch_accuracy == pytest.approx(0.4)
    assert s.wasted_prefetch_seconds == pytest.approx(0.6)


def test_resident_rate_counts_both_tiers_that_avoid_the_disk():
    s = PolicyStats(lookups=100, vram_hits=60, ram_hits=30, nvme_reads=10)
    assert s.resident_rate == pytest.approx(0.9)
    assert s.vram_hit_rate == pytest.approx(0.6)

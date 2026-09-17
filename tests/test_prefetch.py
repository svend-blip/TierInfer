"""Tests for speculative expert loading.

The property that matters most is not speed: it is that a wrong guess costs
bandwidth and never correctness. Routing is the authority, and an expert it
names is always delivered — from the cache, from a prefetch, or from an exact
read issued then and there.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.cache import ExpertCache  # noqa: E402
from tierinfer.index import ByteRange  # noqa: E402
from tierinfer.predict import Frequency, Predictor, Prediction  # noqa: E402
from tierinfer.prefetch import Prefetcher  # noqa: E402
from tierinfer.storage import StorageBackend  # noqa: E402
from tierinfer.stream import BufferPool, ExpertStreamer  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402

EXPERT = 4096
LAYERS, EXPERTS = 4, 8


@pytest.fixture
def modelfile(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(bytes((i * 31 + 7) % 251 for i in range(LAYERS * EXPERTS * EXPERT)))
    return p


def ranges_for(key):
    layer, expert = key
    off = (layer * EXPERTS + expert) * EXPERT
    return [ByteRange(name=f"l{layer}e{expert}", file_offset=off, nbytes=EXPERT)]


@pytest.fixture
def parts(modelfile):
    backend = StorageBackend(modelfile)
    streamer = ExpertStreamer(backend, BufferPool(EXPERT, 8), workers=2)
    tracker = ExpertTracker()
    cache = ExpertCache(16 * EXPERT, tracker)
    yield backend, streamer, cache, tracker
    streamer.close()
    backend.close()


class Fixed(Predictor):
    """Predicts exactly what it is told to, so tests control the guess."""

    name = "fixed"

    def __init__(self, experts):
        self.experts = list(experts)

    def observe(self, routing):
        pass

    def predict(self, layer, sofar=None):
        n = len(self.experts) or 1
        return Prediction(layer, tuple(self.experts),
                          tuple(1.0 / n for _ in self.experts))

    def score(self, layer, sofar):
        return {e: 1.0 for e in self.experts}


# -- the guarantee ------------------------------------------------------


def test_an_expert_nobody_predicted_is_still_delivered(parts, modelfile):
    """The whole safety argument: a miss falls back to an exact load."""
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(0, {})
    got = p.on_routing(0, [5])                     # 5 was never predicted
    want = modelfile.read_bytes()[5 * EXPERT:6 * EXPERT]
    assert got[(0, 5)] == want
    assert p.stats.stalls == 1
    assert p.stats.exact_fallbacks == 1


def test_every_routed_expert_comes_back_however_wrong_the_guess(parts, modelfile):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([7, 6]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(1, {})
    routed = [0, 1, 2]
    got = p.on_routing(1, routed)
    data = modelfile.read_bytes()
    for e in routed:
        off = (1 * EXPERTS + e) * EXPERT
        assert got[(1, e)] == data[off:off + EXPERT]


def test_a_prefetch_that_was_right_avoids_the_stall(parts, modelfile):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([3]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(2, {})
    got = p.on_routing(2, [3])
    off = (2 * EXPERTS + 3) * EXPERT
    assert got[(2, 3)] == modelfile.read_bytes()[off:off + EXPERT]
    assert p.stats.stalls == 0
    assert p.stats.stalls_avoided == 1
    assert p.stats.used == 1


def test_a_failed_prefetch_falls_back_rather_than_failing_the_token(parts, modelfile):
    """A speculative read that errors must not become a correctness problem."""
    _, streamer, cache, tracker = parts

    bad = {(0, 2): [ByteRange("past the end", modelfile.stat().st_size * 4, EXPERT)]}
    p = Prefetcher(streamer, cache, Fixed([2]), 
                   lambda k: bad.get(k, ranges_for(k)), tracker=tracker, depth=4)
    p.before_layer(0, {})
    bad.clear()                                     # the exact path sees good ranges
    got = p.on_routing(0, [2])
    off = 2 * EXPERT
    assert got[(0, 2)] == modelfile.read_bytes()[off:off + EXPERT]
    assert p.stats.stalls == 1, "a failed prefetch is a stall from the token's view"


# -- what it costs ------------------------------------------------------


def test_accuracy_counts_prefetches_used_over_prefetches_issued(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0, 1, 2, 3]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(0, {})
    p.on_routing(0, [0, 1])
    p.end_token()
    assert p.stats.issued == 4
    assert p.stats.used == 2
    assert p.stats.accuracy == pytest.approx(0.5)


def test_unused_prefetches_are_counted_as_wasted_bandwidth(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0, 1, 2, 3]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(0, {})
    p.on_routing(0, [0])
    p.end_token()
    assert p.stats.wasted_bytes == 3 * EXPERT
    assert p.stats.cancelled == 3


def test_lead_time_is_recorded_for_prefetches_that_were_used(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([1]), ranges_for, tracker=tracker, depth=2)
    p.before_layer(0, {})
    p.on_routing(0, [1])
    assert p.stats.lead_count == 1
    assert p.stats.mean_lead_seconds >= 0.0


def test_the_stall_rate_reflects_both_outcomes(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0]), ranges_for, tracker=tracker, depth=2)
    p.before_layer(0, {})
    p.on_routing(0, [0, 1])                        # one hit, one miss
    assert p.stats.stall_rate == pytest.approx(0.5)


# -- not doing unnecessary work -----------------------------------------


def test_a_resident_expert_is_not_prefetched_again(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0, 1]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(0, {})
    p.on_routing(0, [0, 1])
    p.end_token()
    before = p.stats.issued
    p.before_layer(0, {})                          # both are in the cache now
    assert p.stats.issued == before


def test_an_expert_already_in_flight_is_not_submitted_twice(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0, 1]), ranges_for, tracker=tracker, depth=4)
    first = p.before_layer(0, {})
    second = p.before_layer(0, {})
    assert len(first) == 2 and second == []


def test_running_out_of_buffers_stops_speculation_rather_than_raising(parts):
    backend, _, cache, tracker = parts
    small = ExpertStreamer(backend, BufferPool(EXPERT, 2), workers=1)
    try:
        p = Prefetcher(small, cache, Fixed([0, 1, 2, 3, 4]), ranges_for,
                       tracker=tracker, depth=8)
        issued = p.before_layer(0, {})
        assert len(issued) == 2
        assert p.stats.pool_exhausted == 1
        p.on_routing(0, [0, 1])
        p.end_token()
    finally:
        small.close()


def test_dropping_unused_prefetches_returns_every_buffer(parts):
    backend, streamer, cache, tracker = parts
    pool = streamer.pool
    p = Prefetcher(streamer, cache, Fixed([0, 1, 2, 3]), ranges_for, tracker=tracker, depth=4)
    p.before_layer(0, {})
    p.on_routing(0, [0])
    p.end_token()
    assert pool.in_use == 0, "a token boundary leaked buffers"


def test_the_token_boundary_advances_the_trackers_window(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0]), ranges_for, tracker=tracker, depth=2)
    p.before_layer(0, {})
    p.on_routing(0, [0])
    before = tracker.token
    p.end_token()
    assert tracker.token == before + 1


def test_confidence_reaches_the_cache_so_it_can_weigh_it(parts):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Fixed([0, 1]), ranges_for, tracker=tracker, depth=2)
    p.before_layer(0, {})
    assert cache._confidence, "the cache was told nothing about the prediction"


def test_a_predictor_with_no_opinion_issues_nothing_and_still_works(parts, modelfile):
    _, streamer, cache, tracker = parts
    p = Prefetcher(streamer, cache, Frequency(), ranges_for, tracker=tracker, depth=4)
    assert p.before_layer(0, {}) == []
    got = p.on_routing(0, [3])
    off = 3 * EXPERT
    assert got[(0, 3)] == modelfile.read_bytes()[off:off + EXPERT]
    assert p.stats.stalls == 1

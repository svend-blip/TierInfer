"""Tests for speculative expert loading.

The property that matters most is not speed: it is that a wrong guess costs
bandwidth and never correctness. Routing is the authority, and an expert it
names is always delivered — from the cache, from a prefetch, or from an exact
read issued then and there.
"""

from __future__ import annotations

import os
import sys
import threading
import time

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
    # A token boundary no longer waits for a read that is mid-flight: it
    # orphans the load and reaps its buffer once the worker is done. So the
    # buffers are all back *soon*, not *now* — and never leaked.
    deadline = time.perf_counter() + 5
    while (pool.in_use or p.orphans) and time.perf_counter() < deadline:
        time.sleep(0.005)
        p._reap()
    assert p.orphans == 0
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


# -- added 2026-09-18 with the audit's repairs ---------------------------


def _rig(modelfile, predictor, depth=2, slots=8):
    b = StorageBackend(modelfile)
    s = ExpertStreamer(b, BufferPool(EXPERT, slots), workers=2)
    t = ExpertTracker()
    c = ExpertCache(64 * EXPERT, t)
    return b, s, t, Prefetcher(s, c, predictor, ranges_for, tracker=t, depth=depth)


class _Fixed(Predictor):
    name = "fixed"

    def __init__(self, experts):
        self.experts = list(experts)

    def observe(self, routing):
        pass

    def score(self, layer, sofar):
        return {e: 1.0 for e in self.experts}


def test_the_tracker_hears_about_a_token_once(modelfile):
    """One entry per token, holding everything the token routed to — not one
    entry per expert admitted, which is what the first version recorded."""
    b, s, t, p = _rig(modelfile, _Fixed([0, 1]))
    with b, s:
        p.on_routing(0, [0, 1, 2])
        p.on_routing(1, [3, 4])
        assert t.tokens_seen == 0
        p.end_token()
        assert t.tokens_seen == 1
        assert t.last_token_experts() == {(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)}
        assert t.activation_rate((0, 2)) == 1.0


def test_a_speculative_range_that_cannot_be_resolved_is_skipped_and_counted(modelfile):
    def flaky(key):
        if key == (0, 1):
            raise KeyError("no such expert")
        return ranges_for(key)
    b = StorageBackend(modelfile)
    s = ExpertStreamer(b, BufferPool(EXPERT, 8), workers=2)
    p = Prefetcher(s, ExpertCache(64 * EXPERT), _Fixed([1, 2]), flaky, depth=2)
    with b, s:
        issued = p.before_layer(0, {})
        assert issued == [(0, 2)]
        assert p.stats.unmappable == 1
        # the routed path does not skip: the caller hears about it
        with pytest.raises(KeyError):
            p.on_routing(0, [1])


def test_dropping_speculation_never_waits_and_leaks_nothing(modelfile):
    """A load mid-read is orphaned, not waited on; its buffer comes back later."""
    b, s, t, p = _rig(modelfile, _Fixed(list(range(8))), depth=8, slots=8)
    with b, s:
        p.before_layer(0, {})
        t0 = time.perf_counter()
        dropped = p.drop_unused()
        assert time.perf_counter() - t0 < 1.0
        assert dropped == 8
        deadline = time.perf_counter() + 5
        while p.orphans and time.perf_counter() < deadline:
            time.sleep(0.01)
            p._reap()
        assert p.orphans == 0
        assert s.pool.in_use == 0


def test_a_prefetch_that_had_not_landed_is_counted_late(modelfile):
    class Slow(StorageBackend):
        def fd_for(self, r):
            if threading.current_thread().name.startswith("tierinfer-stream"):
                time.sleep(0.2)
            return super().fd_for(r)
    b = Slow(modelfile)
    s = ExpertStreamer(b, BufferPool(EXPERT, 8), workers=2)
    p = Prefetcher(s, ExpertCache(64 * EXPERT), _Fixed([0]), ranges_for, depth=1)
    with b, s:
        p.before_layer(0, {})
        out = p.on_routing(0, [0])              # arrives while the read is sleeping
        assert out[(0, 0)] == modelfile.read_bytes()[:EXPERT]
        assert p.stats.used == 1 and p.stats.late == 1 and p.stats.useful == 0
        assert p.stats.late_wait_seconds > 0.1
        assert p.stats.stalls == 0


def test_a_wait_timeout_falls_back_to_the_exact_path(modelfile):
    class Stuck(StorageBackend):
        def fd_for(self, r):
            if threading.current_thread().name.startswith("tierinfer-stream"):
                time.sleep(0.5)
            return super().fd_for(r)
    b = Stuck(modelfile)
    s = ExpertStreamer(b, BufferPool(EXPERT, 8), workers=2)
    p = Prefetcher(s, ExpertCache(64 * EXPERT), _Fixed([0]), ranges_for, depth=1,
                   wait_timeout=0.05)
    with b, s:
        p.before_layer(0, {})
        out = p.on_routing(0, [0])
        assert out[(0, 0)] == modelfile.read_bytes()[:EXPERT]
        assert p.stats.timed_out == 1 and p.stats.exact_fallbacks == 1
        assert p.orphans == 1
        time.sleep(0.6)
        p._reap()
        assert p.orphans == 0 and s.pool.in_use == 0


def test_the_cache_hears_what_each_expert_cost_to_load(modelfile):
    b, s, t, p = _rig(modelfile, _Fixed([0]), depth=1)
    with b, s:
        p.before_layer(0, {})
        p.on_routing(0, [0, 5])         # 0 prefetched, 5 an exact read
        for key in ((0, 0), (0, 5)):
            st = t.stats[key]
            assert st.loads == 1 and st.load_seconds > 0

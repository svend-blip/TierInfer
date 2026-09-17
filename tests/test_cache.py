"""Tests for the bounded expert cache and its eviction policy."""

from __future__ import annotations

import pytest

from tierinfer.cache import ExpertCache
from tierinfer.tracker import ExpertTracker

MB = 1024 * 1024


def _tracker(routes):
    t = ExpertTracker()
    for r in routes:
        t.begin_token()
        t.record(r)
    return t


def test_a_hit_returns_the_payload_and_a_miss_returns_none():
    c = ExpertCache(10 * MB)
    c.put((0, 1), b"weights", MB)
    assert c.get((0, 1)) == b"weights"
    assert c.get((0, 2)) is None
    assert c.stats.hits == 1 and c.stats.misses == 1
    assert c.stats.hit_rate == 0.5


def test_capacity_is_bytes_not_entries():
    c = ExpertCache(3 * MB)
    assert c.put((0, 1), None, 2 * MB)
    assert c.put((0, 2), None, 1 * MB)
    assert c.used_bytes == 3 * MB and len(c) == 2
    c.put((0, 3), None, 1 * MB)
    assert c.used_bytes <= 3 * MB


def test_the_frequently_used_expert_survives_the_recently_used_one():
    """The reason this is not an LRU: recency alone gets this backwards."""
    t = _tracker([{(0, 1)}] * 8 + [{(0, 2)}])
    c = ExpertCache(2 * MB, t)
    c.put((0, 1), None, MB)
    c.put((0, 2), None, MB)
    c.put((0, 3), None, MB)          # forces one eviction
    assert (0, 1) in c               # used by 8 of 9 tokens
    assert (0, 2) not in c           # used once, and more recently


def test_a_pinned_entry_is_never_evicted():
    t = _tracker([{(0, 9)}] * 5)
    c = ExpertCache(2 * MB, t)
    c.put((0, 1), None, MB, pinned=True)
    c.put((0, 2), None, MB)
    c.put((0, 3), None, MB)
    assert (0, 1) in c


def test_everything_pinned_means_a_new_entry_is_refused_not_forced_in():
    c = ExpertCache(2 * MB)
    c.put((0, 1), None, MB, pinned=True)
    c.put((0, 2), None, MB, pinned=True)
    assert c.put((0, 3), None, MB) is False
    assert c.used_bytes == 2 * MB


def test_an_entry_larger_than_the_cache_is_refused():
    c = ExpertCache(MB)
    assert c.put((0, 1), None, 4 * MB) is False
    assert c.stats.rejected_oversize == 1
    assert len(c) == 0


def test_prediction_confidence_protects_an_expert_with_no_history():
    t = _tracker([{(0, 1)}] * 4)
    c = ExpertCache(2 * MB, t)
    c.put((0, 1), None, MB)
    c.put((0, 7), None, MB)
    c.set_confidence({(0, 7): 1.0})
    c.put((0, 8), None, MB)
    assert (0, 7) in c


def test_a_large_expert_must_outweigh_the_small_ones_it_displaces():
    t = _tracker([{(0, 1), (0, 2)}] * 4)
    c = ExpertCache(4 * MB, t)
    c.put((0, 1), None, MB)
    c.put((0, 2), None, 3 * MB)
    # equal activation rates, so value is decided by size: the 3 MB entry is
    # worth less per byte and goes first.
    assert c._choose_victim() == (0, 2)


def test_would_evict_for_names_the_cost_before_paying_it():
    t = _tracker([{(0, 1)}] * 5)
    c = ExpertCache(3 * MB, t)
    c.put((0, 1), None, MB)
    c.put((0, 2), None, MB)
    c.put((0, 3), None, MB)
    victims = c.would_evict_for(2 * MB)
    assert len(victims) == 2
    assert (0, 1) not in victims       # the hot one is not among them
    assert len(c) == 3                 # and nothing actually moved


def test_missing_reports_without_counting_as_a_lookup():
    c = ExpertCache(2 * MB)
    c.put((0, 1), None, MB)
    assert c.missing([(0, 1), (0, 2)]) == [(0, 2)]
    assert c.stats.lookups == 0


def test_reinserting_a_key_replaces_rather_than_doubles_its_bytes():
    c = ExpertCache(4 * MB)
    c.put((0, 1), None, MB)
    c.put((0, 1), None, 2 * MB)
    assert c.used_bytes == 2 * MB and len(c) == 1


def test_a_zero_capacity_cache_is_refused():
    with pytest.raises(ValueError):
        ExpertCache(0)


# -- the heap, and the risks a lazy heap carries -------------------------


def test_the_heap_picks_what_the_exact_scan_would_when_nothing_has_drifted():
    t = _tracker([{(0, 1)}] * 6 + [{(0, 2)}] * 2 + [{(0, 3)}])
    c = ExpertCache(4 * MB, t)
    for k in ((0, 1), (0, 2), (0, 3)):
        c.put(k, None, MB)
    assert c._choose_victim() == c._scan_victim()


def test_a_stale_entry_whose_value_rose_is_re_pushed_rather_than_evicted():
    """The failure a naive heap would have: evicting on a score that has aged."""
    t = ExpertTracker()
    c = ExpertCache(3 * MB, t)
    for k in ((0, 1), (0, 2)):
        t.begin_token(); t.record({k})
        c.put(k, None, MB)
    # (0, 1) was cheap when pushed; now make it the hottest thing in the cache.
    for _ in range(20):
        t.begin_token(); t.record({(0, 1)})
    assert c._choose_victim() == (0, 2)
    assert c._choose_victim() == c._scan_victim()   # and the heap agrees with the scan


def test_an_entry_removed_from_the_cache_is_skipped_on_the_way_out():
    t = _tracker([{(0, 1)}] * 3)
    c = ExpertCache(4 * MB, t)
    for k in ((0, 1), (0, 2), (0, 3)):
        c.put(k, None, MB)
    c._drop((0, 2), evicted=False)      # leaves a stale heap entry behind
    victim = c._choose_victim()
    assert victim in ((0, 1), (0, 3))


def test_pinning_after_insertion_is_honoured_by_the_heap():
    t = _tracker([{(0, 9)}] * 5)
    c = ExpertCache(2 * MB, t)
    c.put((0, 1), None, MB)
    c.put((0, 2), None, MB)
    c.pin((0, 1))                       # pinned after it was already pushed
    assert c._choose_victim() == (0, 2)
    c.put((0, 3), None, MB)
    assert (0, 1) in c


def test_raising_confidence_re_pushes_only_the_named_experts():
    t = _tracker([{(0, 1)}] * 4)
    c = ExpertCache(3 * MB, t)
    for k in ((0, 1), (0, 2), (0, 3)):
        c.put(k, None, MB)
    before = len(c._heap)
    c.set_confidence({(0, 2): 1.0})
    assert len(c._heap) == before + 1
    assert c._choose_victim() != (0, 2)


def test_eviction_stays_cheap_as_the_cache_grows():
    """The defect this replaced: an O(n) scan, over a millisecond at 3 000 entries."""
    import time

    def cost(n):
        t = ExpertTracker(window=64)
        c = ExpertCache(n * MB, t)
        for i in range(n):
            c.put((0, i), None, MB)
        start = time.perf_counter()
        for i in range(100):
            c.put((1, i), None, MB)
        return (time.perf_counter() - start) / 100

    small, large = cost(100), cost(2000)
    # Twenty times the entries must not cost anything like twenty times the time.
    assert large < small * 5


def test_asking_for_a_victim_does_not_remove_it_from_the_running():
    """The trap in a popping heap: a victim the caller declines is lost for good."""
    t = _tracker([{(0, 1)}] * 4)
    c = ExpertCache(4 * MB, t)
    for k in ((0, 1), (0, 2), (0, 3)):
        c.put(k, None, MB)
    first = c._choose_victim()
    assert c._choose_victim() == first          # asking twice answers twice
    assert first == c._scan_victim()
    assert c.heap_fallbacks == 0


def test_a_scan_fallback_is_always_counted():
    t = _tracker([{(0, 1)}] * 3)
    c = ExpertCache(3 * MB, t)
    for k in ((0, 1), (0, 2), (0, 3)):
        c.put(k, None, MB)
    c._heap.clear()                             # simulate a drained heap
    assert c._choose_victim() == c._scan_victim()
    assert c.heap_fallbacks == 1

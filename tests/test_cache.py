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

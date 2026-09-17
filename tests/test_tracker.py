"""Tests for the activity tracker."""

from __future__ import annotations

import pytest

from tierinfer.tracker import ExpertTracker


def _run(tracker, token_routes):
    for route in token_routes:
        tracker.begin_token()
        tracker.record(route)


def test_activation_rate_is_the_share_of_tokens_that_used_an_expert():
    t = ExpertTracker()
    _run(t, [{(0, 1)}, {(0, 1)}, {(0, 2)}, {(0, 1)}])
    assert t.activation_rate((0, 1)) == 0.75
    assert t.activation_rate((0, 2)) == 0.25
    assert t.activation_rate((0, 9)) == 0.0


def test_recency_counts_tokens_since_last_use_and_marks_the_unseen():
    t = ExpertTracker()
    _run(t, [{(0, 1)}, {(0, 2)}, {(0, 2)}])
    assert t.recency((0, 1)) == 2
    assert t.recency((0, 2)) == 0
    assert t.recency((0, 5)) == -1


def test_reuse_distance_is_absent_rather_than_large_for_a_single_use():
    """An expert seen once has unknown reuse, which is not the same as never."""
    t = ExpertTracker()
    _run(t, [{(0, 1)}])
    assert t.stats[(0, 1)].mean_reuse_distance == 0.0
    assert not t.stats[(0, 1)].reuse_distances


def test_reuse_distance_averages_the_gaps_between_uses():
    t = ExpertTracker()
    _run(t, [{(0, 1)}, set(), {(0, 1)}, set(), set(), {(0, 1)}])
    assert list(t.stats[(0, 1)].reuse_distances) == [2, 3]
    assert t.stats[(0, 1)].mean_reuse_distance == pytest.approx(2.5)


def test_the_window_bounds_what_the_rate_is_measured_over():
    t = ExpertTracker(window=4)
    _run(t, [{(0, 1)}] * 4 + [{(0, 2)}] * 4)
    assert t.activation_rate((0, 1)) == 0.0
    assert t.activation_rate((0, 2)) == 1.0
    assert t.tokens_seen == 4


def test_hot_ranks_by_rate_then_recency():
    t = ExpertTracker()
    _run(t, [{(0, 1), (0, 2)}, {(0, 1)}, {(0, 3)}])
    assert t.hot(1) == [(0, 1)]
    assert set(t.hot(3)) == {(0, 1), (0, 2), (0, 3)}


def test_coverage_grades_a_guess_against_what_the_token_needed():
    t = ExpertTracker()
    _run(t, [{(0, 1), (0, 2), (0, 3), (0, 4)}])
    assert t.coverage({(0, 1), (0, 2)}) == 0.5
    assert t.coverage({(0, 1), (0, 2), (0, 3), (0, 4)}) == 1.0
    assert t.coverage({(9, 9)}) == 0.0


def test_load_cost_is_averaged_per_expert():
    t = ExpertTracker()
    t.record_load((0, 1), 0.010)
    t.record_load((0, 1), 0.020)
    assert t.stats[(0, 1)].mean_load_seconds == pytest.approx(0.015)


def test_a_window_below_one_token_is_refused():
    with pytest.raises(ValueError):
        ExpertTracker(window=0)

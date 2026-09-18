"""The trainable prerouter: learns a deterministic transition, beats the
counting predictor on it, survives a save/load, and refuses the wrong model."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

np = pytest.importorskip("numpy")

from tierinfer.predict import Transition, evaluate  # noqa: E402
from tierinfer.prerouter import Prerouter  # noqa: E402

LAYERS, N, K = [0, 1, 2], 16, 4


def _trace(tokens, seed=1):
    """Layer L+1 routes to (e + 3) mod N of layer L's experts, plus noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(tokens):
        r = {0: sorted(rng.choice(N, K, replace=False).tolist())}
        for l in LAYERS[1:]:
            nxt = [(e + 3) % N for e in r[l - 1]]
            if rng.random() < 0.2:
                nxt[0] = int(rng.integers(N))
            r[l] = sorted(set(nxt))
        out.append(r)
    return out


def test_it_learns_the_transition_and_matches_or_beats_counting():
    train, test = _trace(400, seed=1), _trace(100, seed=2)
    pr = Prerouter(LAYERS, N, lr=0.1)
    losses = pr.train(train, epochs=5)
    assert losses[-1] < losses[0]
    pr.online = False
    tr = Transition()
    for r in train:
        tr.observe(r)
    a = evaluate(pr, test, k=K).recall
    b = evaluate(tr, test, k=K).recall
    assert a > 0.6
    assert a >= b - 0.05, (a, b)


def test_save_and_load_round_trip_and_the_shape_check(tmp_path):
    pr = Prerouter(LAYERS, N)
    pr.train(_trace(50), epochs=1)
    p = tmp_path / "pr.npz"
    pr.save(p)
    back = Prerouter.load(p, expect_layers=LAYERS, expect_experts=N)
    r = _trace(1, seed=9)[0]
    assert back.score(1, {0: r[0]}) == pytest.approx(pr.score(1, {0: r[0]}))
    assert back.trained_tokens == pr.trained_tokens
    with pytest.raises(ValueError):
        Prerouter.load(p, expect_experts=N + 1)
    with pytest.raises(ValueError):
        Prerouter.load(p, expect_layers=[0, 1])


def test_it_keeps_learning_online_when_observed():
    pr = Prerouter(LAYERS, N)
    before = pr.steps
    for r in _trace(5):
        pr.observe(r)
    assert pr.steps > before and pr.trained_tokens == 5

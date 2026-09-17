"""Tests for expert prediction.

The thing these have to protect is the boundary: a predictor ranks, it never
answers. Beyond that they pin the arithmetic of the score, the fallbacks that
keep a cold predictor from returning nothing, and the one property that
decides whether any of this is worth its memory — beating frequency.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.predict import (  # noqa: E402
    Blend, Frequency, Persistence, Prediction, Transition, evaluate,
)


def trace_fixed(n, routing):
    return [dict(routing) for _ in range(n)]


# -- the shape of a prediction ------------------------------------------


def test_a_prediction_ranks_by_score_and_normalises_confidence():
    f = Frequency()
    for _ in range(3):
        f.observe({0: [7]})
    f.observe({0: [2]})
    p = f.predict(0)
    assert p.experts[0] == 7
    assert p.confidence[0] == pytest.approx(0.75)
    assert sum(p.confidence) == pytest.approx(1.0)


def test_ties_break_on_expert_id_so_runs_are_reproducible():
    f = Frequency()
    f.observe({0: [5, 3, 9]})
    assert f.predict(0).experts == (3, 5, 9)


def test_an_untrained_predictor_returns_an_empty_prediction_not_a_guess():
    p = Frequency().predict(0)
    assert p.experts == () and p.confidence == ()
    assert p.top(4) == ()


def test_top_k_never_exceeds_what_is_known():
    f = Frequency()
    f.observe({0: [1, 2]})
    assert len(f.predict(0).top(10)) == 2


def test_scores_can_be_read_as_a_mapping():
    f = Frequency()
    f.observe({0: [1, 1, 2]})
    assert f.predict(0).as_scores()[1] > f.predict(0).as_scores()[2]


# -- each predictor's own claim -----------------------------------------


def test_frequency_ignores_context():
    f = Frequency()
    f.observe({0: [1], 1: [9]})
    assert f.score(0, {1: [9]}) == f.score(0, {})


def test_persistence_prefers_the_previous_token_over_the_common_one():
    p = Persistence()
    for _ in range(20):
        p.observe({0: [1]})          # 1 is overwhelmingly the frequent expert
    p.observe({0: [4]})              # but 4 was last
    assert p.predict(0).experts[0] == 4


def test_persistence_decays_over_its_depth():
    p = Persistence(decay=0.5, depth=3)
    p.observe({0: [1]}); p.observe({0: [2]}); p.observe({0: [3]})
    s = p.score(0, {})
    assert s[3] > s[2] > s[1]


def test_persistence_falls_back_rather_than_returning_nothing():
    p = Persistence()
    p.observe({0: [1, 2]})
    assert p.score(5, {}) == {}          # nothing known about layer 5 at all
    p.observe({5: [8]})
    assert p.score(5, {})


def test_persistence_refuses_a_decay_that_is_not_a_decay():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            Persistence(decay=bad)


def test_transition_uses_this_tokens_earlier_layers():
    t = Transition()
    for _ in range(10):
        t.observe({0: [1], 1: [11]})
        t.observe({0: [2], 1: [22]})
    assert t.predict(1, {0: [1]}).experts[0] == 11
    assert t.predict(1, {0: [2]}).experts[0] == 22


def test_transition_without_a_previous_layer_falls_back_to_frequency():
    t = Transition()
    for _ in range(5):
        t.observe({0: [3], 1: [7]})
    assert t.predict(0, {}).experts[0] == 3      # layer 0 has no predecessor


def test_transition_falls_back_when_the_prior_state_was_never_seen():
    t = Transition()
    for _ in range(5):
        t.observe({0: [1], 1: [11]})
    got = t.predict(1, {0: [99]})                # 99 never preceded anything
    assert got.experts, "an unseen prior state must not empty the prediction"


# -- the blend ----------------------------------------------------------

def test_a_blend_normalises_before_weighting():
    """Raw transition counts dwarf persistence weights; unnormalised, the
    blend would just be transition under another name."""
    freq, pers = Frequency(), Persistence()
    b = Blend([(freq, 0.5), (pers, 0.5)])
    for _ in range(100):
        b.observe({0: [1]})
    b.observe({0: [2]})
    s = b.score(0, {})
    assert s[2] > 0.2, "the low-count part was drowned out"


def test_a_shared_sub_predictor_learns_once_not_twice():
    f = Frequency()
    b = Blend([(f, 0.5), (f, 0.5)])
    b.observe({0: [1]})
    assert f.counts[0][1] == 1


def test_a_blend_needs_parts_and_sane_weights():
    with pytest.raises(ValueError):
        Blend([])
    with pytest.raises(ValueError):
        Blend([(Frequency(), -1.0)])


def test_a_part_that_knows_nothing_does_not_poison_the_blend():
    known = Frequency()
    known.observe({0: [5]})
    b = Blend([(known, 1.0), (Frequency(), 1.0)])
    assert b.predict(0).experts[0] == 5


# -- the scoring harness ------------------------------------------------


def test_recall_counts_used_experts_found_not_predictions_made():
    f = Frequency()
    trace = trace_fixed(10, {0: [1, 2]})
    s = evaluate(f, trace, k=4, warmup=1)
    assert s.tokens == 9
    assert s.used == 18
    assert s.recall == pytest.approx(1.0)


def test_waste_is_reported_alongside_recall():
    f = Frequency()
    s = evaluate(f, trace_fixed(10, {0: [1, 2]}), k=4, warmup=1)
    # Only 2 experts exist, so k=4 can predict at most 2: nothing is wasted.
    assert s.predicted == 18
    assert s.wasted == pytest.approx(0.0)


def test_a_predictor_that_knows_nothing_scores_zero_not_an_error():
    class Blind(Frequency):
        name = "blind"

        def score(self, layer, sofar):
            return {}

    s = evaluate(Blind(), trace_fixed(5, {0: [1]}), k=2)
    assert s.recall == 0.0 and s.wasted == 0.0


def test_warmup_tokens_are_learned_from_but_not_scored():
    f = Frequency()
    s = evaluate(f, trace_fixed(10, {0: [1]}), k=1, warmup=4)
    assert s.tokens == 6
    assert f.counts[0][1] == 10, "warmup tokens must still be observed"


def test_the_evaluation_never_shows_a_layer_its_own_answer():
    """If the harness leaked the current layer's routing, recall would be 1.0."""
    seen = []

    class Spy(Frequency):
        name = "spy"

        def score(self, layer, sofar):
            seen.append((layer, dict(sofar)))
            return super().score(layer, sofar)

    evaluate(Spy(), [{0: [1], 1: [2], 2: [3]}], k=1)
    for layer, sofar in seen:
        assert layer not in sofar, f"layer {layer} saw its own routing"
        assert all(l < layer for l in sofar)


def test_per_layer_recall_finds_the_worst_layers():
    class Half(Frequency):
        name = "half"

        def score(self, layer, sofar):
            return {1: 1.0} if layer == 0 else {99: 1.0}

    s = evaluate(Half(), trace_fixed(4, {0: [1], 1: [1]}), k=1)
    worst = s.worst_layers(1)
    assert worst[0][0] == 1 and worst[0][1] == 0.0


# -- the property that justifies the whole module -----------------------


def test_context_beats_frequency_on_a_trace_where_context_exists():
    """The floor is frequency. A predictor that cannot beat it is not earning
    the memory it costs."""
    rnd = random.Random(20260917)
    experts = list(range(32))
    trace = []
    state = rnd.choice(experts)
    for _ in range(400):
        # Layer 0 wanders slowly; layer 1 is a deterministic function of it.
        if rnd.random() < 0.15:
            state = rnd.choice(experts)
        trace.append({0: [state], 1: [(state * 7 + 3) % 32]})

    freq = evaluate(Frequency(), list(trace), k=4, warmup=50)
    trans = evaluate(Transition(), list(trace), k=4, warmup=50)
    pers = evaluate(Persistence(), list(trace), k=4, warmup=50)

    assert trans.recall > freq.recall + 0.2, (trans.recall, freq.recall)
    assert pers.recall > freq.recall + 0.2, (pers.recall, freq.recall)


def test_prediction_is_never_asked_to_decide_what_is_used():
    """The boundary, pinned as an interface fact: nothing here returns a
    routing decision, only rankings and confidences."""
    for p in (Frequency(), Persistence(), Transition(),
              Blend([(Frequency(), 1.0)])):
        p.observe({0: [1, 2]})
        got = p.predict(0)
        assert isinstance(got, Prediction)
        assert not hasattr(got, "route") and not hasattr(got, "decide")
        assert set(got.experts) <= {1, 2}


# -- the adaptive blend -------------------------------------------------


def test_adaptive_starts_with_no_opinion():
    from tierinfer.predict import AdaptiveBlend
    a = AdaptiveBlend([Frequency(), Persistence(), Transition()], k=4)
    w = a.weights()
    assert len(set(round(v, 6) for v in w.values())) == 1, "it must not favour a part before measuring"


def test_adaptive_moves_its_weight_onto_whichever_part_is_winning():
    """The point of measuring is being able to act on it.

    Which part wins is not asserted here — it is measured. An earlier version
    of this test named transition and failed against correct code, because on
    a trace whose state changes one token in ten the strongest signal is
    simply "the same experts as last token".
    """
    from tierinfer.predict import AdaptiveBlend
    rnd = random.Random(7)
    trace = []
    state = 0
    for _ in range(300):
        if rnd.random() < 0.1:
            state = rnd.randrange(16)
        trace.append({0: [state], 1: [(state * 5 + 1) % 16]})

    standalone = {p.name: evaluate(p, list(trace), k=2, warmup=40).recall
                  for p in (Frequency(), Persistence(), Transition())}
    best = max(standalone, key=standalone.get)

    a = AdaptiveBlend([Frequency(), Persistence(), Transition()], k=2, update_every=1)
    evaluate(a, list(trace), k=2, warmup=40)
    w = a.weights()
    assert max(w, key=w.get) == best, (w, standalone)
    assert w[best] > 0.5, w


def test_adaptive_is_not_dragged_below_its_best_part():
    from tierinfer.predict import AdaptiveBlend
    rnd = random.Random(11)
    trace = []
    state = 0
    for _ in range(300):
        if rnd.random() < 0.1:
            state = rnd.randrange(16)
        trace.append({0: [state], 1: [(state * 5 + 1) % 16]})

    parts = [Frequency(), Persistence(), Transition()]
    adaptive = evaluate(AdaptiveBlend([Frequency(), Persistence(), Transition()],
                                      k=4, update_every=1), list(trace), k=4, warmup=40)
    best = max(evaluate(p, list(trace), k=4, warmup=40).recall for p in parts)
    assert adaptive.recall > best - 0.10, (adaptive.recall, best)


def test_a_part_that_never_wins_is_kept_at_the_floor_not_deleted():
    """Which signal wins changes with the prompt, so nothing is discarded."""
    from tierinfer.predict import AdaptiveBlend
    a = AdaptiveBlend([Frequency(), Transition()], k=2, floor=0.02)
    a.recall[id(a.parts[0])] = 0.9
    a.recall[id(a.parts[1])] = 0.0
    w = a.weights()
    assert w["transition"] > 0
    assert w["frequency"] > w["transition"] * 10


def test_adaptive_refuses_settings_that_would_flatten_the_measurement():
    from tierinfer.predict import AdaptiveBlend
    for kwargs in ({"alpha": 0.0}, {"alpha": 1.5}, {"update_every": 0}, {"sharpness": 0.5}):
        with pytest.raises(ValueError):
            AdaptiveBlend([Frequency()], **kwargs)
    with pytest.raises(ValueError):
        AdaptiveBlend([])

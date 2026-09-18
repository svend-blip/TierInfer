"""Tests for the one invariant: a routed expert is always delivered.

These are adversarial on purpose. Each one breaks a different part of the
speculative machinery and asserts the bytes still come back correct, because
a rule that only holds when nothing is wrong is not a rule.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.cache import ExpertCache  # noqa: E402
from tierinfer.index import ByteRange  # noqa: E402
from tierinfer.prefetch import Prefetcher  # noqa: E402
from tierinfer.predict import Frequency, Prediction, Predictor  # noqa: E402
from tierinfer.safety import (  # noqa: E402
    AuditFinding, GuardStats, SPECULATIVE_MODULES, SafetyError, audit,
    exact_load, guard,
)
from tierinfer.storage import StorageBackend  # noqa: E402
from tierinfer.stream import BufferPool, ExpertStreamer  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402

EXPERT = 4096
LAYERS, EXPERTS = 4, 8


@pytest.fixture
def modelfile(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(bytes((i * 37 + 11) % 251 for i in range(LAYERS * EXPERTS * EXPERT)))
    return p


@pytest.fixture
def backend(modelfile):
    with StorageBackend(modelfile) as b:
        yield b


def ranges_for(key):
    layer, expert = key
    off = (layer * EXPERTS + expert) * EXPERT
    return [ByteRange(name=f"l{layer}e{expert}", file_offset=off, nbytes=EXPERT)]


def truth(modelfile, key):
    layer, expert = key
    off = (layer * EXPERTS + expert) * EXPERT
    return modelfile.read_bytes()[off:off + EXPERT]


# -- the bottom of the stack -------------------------------------------


def test_the_exact_path_returns_the_files_bytes(backend, modelfile):
    assert exact_load(backend, ranges_for((2, 3))) == truth(modelfile, (2, 3))


def test_the_exact_path_refuses_to_be_asked_for_nothing(backend):
    with pytest.raises(SafetyError, match="no byte ranges"):
        exact_load(backend, [])


def test_a_failure_in_the_exact_path_is_fatal_rather_than_papered_over(backend,
                                                                      modelfile):
    """There is nothing below this, and inventing a fallback would mean
    returning weights that are not the model's."""
    past_end = modelfile.stat().st_size * 4
    with pytest.raises(SafetyError):
        exact_load(backend, [ByteRange("gone", past_end, EXPERT)])


# -- every way a guess can be wrong ------------------------------------


@pytest.mark.parametrize("broken,reason", [
    (lambda key: None, "absent"),
    (lambda key: (_ for _ in ()).throw(RuntimeError("boom")), "raised"),
    (lambda key: b"short", "wrong size"),
    (lambda key: b"x" * (EXPERT * 2), "wrong size"),
])
def test_a_wrong_guess_becomes_an_exact_load(backend, modelfile, broken, reason):
    stats = GuardStats()
    load = guard(broken, backend, ranges_for, stats)
    assert load((1, 4)) == truth(modelfile, (1, 4))
    assert stats.fallbacks == 1
    assert stats.reasons == {reason: 1}


def test_a_right_guess_is_taken_without_a_read(backend, modelfile):
    stats = GuardStats()
    load = guard(lambda key: truth(modelfile, key), backend, ranges_for, stats)
    assert load((0, 1)) == truth(modelfile, (0, 1))
    assert stats.speculative_hits == 1 and stats.fallbacks == 0


def test_the_reasons_are_counted_apart(backend, modelfile):
    calls = {"n": 0}

    def flaky(key):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("boom")
        if calls["n"] % 3 == 1:
            return None
        return b"short"

    stats = GuardStats()
    load = guard(flaky, backend, ranges_for, stats)
    for i in range(9):
        assert load((0, i % 8)) == truth(modelfile, (0, i % 8))
    assert set(stats.reasons) == {"absent", "raised", "wrong size"}
    assert stats.fallback_rate == 1.0


def test_a_guard_with_no_stats_still_works(backend, modelfile):
    load = guard(lambda key: None, backend, ranges_for)
    assert load((3, 7)) == truth(modelfile, (3, 7))
    assert load.stats.fallbacks == 1


# -- the whole speculative stack, with faults thrown in ----------------


def test_the_prefetcher_delivers_every_expert_under_random_failure(backend,
                                                                   modelfile):
    """Random wrong predictions, random broken byte ranges, and every routed
    expert still comes back byte-correct."""
    rnd = random.Random(20260917)
    tracker = ExpertTracker()
    cache = ExpertCache(8 * EXPERT, tracker)
    pool = BufferPool(EXPERT, 4)
    streamer = ExpertStreamer(backend, pool, workers=2)

    class Erratic(Predictor):
        name = "erratic"

        def observe(self, routing):
            pass

        def predict(self, layer, sofar=None):
            picks = tuple(rnd.randrange(EXPERTS) for _ in range(3))
            return Prediction(layer, picks, tuple(1 / 3 for _ in picks))

        def score(self, layer, sofar):
            return {}

    # Broken only while speculating. A byte range that is wrong on the exact
    # path too is a broken index, not a prediction miss, and goal 15 makes no
    # promise about that — exact_load raises, and there is nothing below it.
    speculating = {"now": True}
    broken = {(1, 2), (3, 5)}

    def sometimes_wrong(key):
        if key in broken and speculating["now"]:
            return [ByteRange("past the end", modelfile.stat().st_size * 8, EXPERT)]
        return ranges_for(key)

    try:
        p = Prefetcher(streamer, cache, Erratic(), sometimes_wrong,
                       tracker=tracker, depth=3)
        for _ in range(20):
            for layer in range(LAYERS):
                speculating["now"] = True
                p.before_layer(layer, {})
                speculating["now"] = False      # the exact path sees the truth
                routed = rnd.sample(range(EXPERTS), 2)
                got = p.on_routing(layer, routed)
                for e in routed:
                    key = (layer, e)
                    assert got[key] == truth(modelfile, key), f"{key} came back wrong"
            p.end_token()
        assert p.stats.exact_fallbacks > 0, "no fallback was exercised"
    finally:
        streamer.close()


def test_a_prefetch_that_failed_is_a_stall_not_a_wrong_answer(backend, modelfile):
    tracker = ExpertTracker()
    cache = ExpertCache(8 * EXPERT, tracker)
    streamer = ExpertStreamer(backend, BufferPool(EXPERT, 4), workers=1)
    bad = {(0, 1): [ByteRange("gone", modelfile.stat().st_size * 8, EXPERT)]}

    class Always(Predictor):
        name = "always"

        def observe(self, routing):
            pass

        def predict(self, layer, sofar=None):
            return Prediction(layer, (1,), (1.0,))

        def score(self, layer, sofar):
            return {1: 1.0}

    try:
        p = Prefetcher(streamer, cache, Always(),
                       lambda k: bad.get(k, ranges_for(k)), tracker=tracker, depth=1)
        p.before_layer(0, {})
        bad.clear()
        assert p.on_routing(0, [1])[(0, 1)] == truth(modelfile, (0, 1))
        assert p.stats.stalls == 1
    finally:
        streamer.close()


# -- the audit ----------------------------------------------------------


def test_every_speculative_module_keeps_a_reachable_exact_path():
    assert audit() == [], [str(f) for f in audit()]


def test_a_module_that_is_not_declared_speculative_is_reported():
    found = audit(["tierinfer.cache"])
    assert found and "SPECULATIVE_MODULES" in str(found[0])


def test_a_module_that_cannot_be_read_is_reported_not_passed():
    found = audit(["tierinfer.nonexistent"])
    assert found


def test_the_audit_names_what_is_missing():
    f = AuditFinding("tierinfer.thing", "load_now")
    assert "load_now" in str(f) and "tierinfer.thing" in str(f)


def test_the_declared_list_covers_the_modules_that_actually_speculate():
    """A speculative module missing from the list is how this invariant would
    be lost: by omission, not by decision.

    Speculating means *using a predictor*, which is a structural fact — an
    import — not a word in a docstring. The first version of this counted
    occurrences of "predict" in the text and flagged `cache` and `trace`,
    both of which only mention prediction in prose. A count cannot read.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "tierinfer"
    speculative = set()
    for path in src.glob("*.py"):
        tree = ast.parse(path.read_text())
        # A module that *defines* a predictor (a class deriving from Predictor)
        # is a source of guesses, not a consumer of them: it moves no bytes, so
        # it has no exact path to keep. The prerouter is one.
        defines = any(isinstance(n, ast.ClassDef) and
                      any(getattr(b, "id", getattr(b, "attr", "")) == "Predictor" for b in n.bases)
                      for n in ast.walk(tree))
        if defines and path.stem != "predict":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in ("predict",
                                                                    "tierinfer.predict"):
                speculative.add(f"tierinfer.{path.stem}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("tierinfer.predict"):
                        speculative.add(f"tierinfer.{path.stem}")
    undeclared = speculative - set(SPECULATIVE_MODULES) - {"tierinfer.safety"}
    assert not undeclared, f"these use a predictor but are not audited: {sorted(undeclared)}"
    assert speculative, "the check found nothing at all, which means it is not working"

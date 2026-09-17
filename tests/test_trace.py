"""Tests for reading a real routing trace.

The trace format is produced by a C++ tool against a real model, so these
tests work from records shaped exactly like the ones that tool emits —
including the ragged shape llama.cpp legitimately produces, which the first
version of the reader rejected as corruption.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.trace import (  # noqa: E402
    TraceError, describe, read_trace, routings,
)


def write(tmp_path, records):
    p = tmp_path / "trace.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


def rec(decode, layer, experts):
    return {"decode": decode, "layer": layer, "n_tokens": len(experts),
            "n_used": len(experts[0]) if experts else 0, "experts": experts}


# -- the ordinary shape -------------------------------------------------


def test_a_prompt_decode_becomes_one_routing_per_token(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1, 2], [3, 4], [5, 6]]),
                         rec(0, 2, [[7, 8], [9, 10], [11, 12]])])
    rows = read_trace(p)
    assert len(rows) == 3
    assert rows[0].routing == {1: (1, 2), 2: (7, 8)}
    assert rows[2].routing == {1: (5, 6), 2: (11, 12)}
    assert all(r.prompt for r in rows)


def test_generated_tokens_are_marked_apart_from_prompt_tokens(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1, 2], [3, 4]]), rec(1, 1, [[5, 6]])])
    rows = read_trace(p)
    assert [r.prompt for r in rows] == [True, True, False]
    assert [r.decode for r in rows] == [0, 0, 1]


def test_either_kind_can_be_asked_for_alone(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1], [2]]), rec(1, 1, [[3]]), rec(2, 1, [[4]])])
    assert len(read_trace(p, generated=False)) == 2
    assert len(read_trace(p, prompt=False)) == 2
    assert len(read_trace(p)) == 4


def test_the_mapping_form_is_what_the_predictors_take(tmp_path):
    p = write(tmp_path, [rec(0, 3, [[1, 2]])])
    got = list(routings(read_trace(p)))
    assert got == [{3: [1, 2]}]


# -- the ragged shape llama.cpp really produces -------------------------


def test_a_final_layer_computed_only_for_the_output_token_is_not_corruption(tmp_path):
    """llama.cpp computes the last layer only for tokens whose logits are
    wanted. The first version of this reader called that a broken trace."""
    p = write(tmp_path, [rec(0, 1, [[1], [2], [3]]),
                         rec(0, 2, [[4], [5], [6]]),
                         rec(0, 3, [[9]])])            # last layer, last token only
    rows = read_trace(p)
    assert len(rows) == 3
    assert rows[0].routing == {1: (1,), 2: (4,)}       # no layer 3
    assert rows[2].routing == {1: (3,), 2: (6,), 3: (9,)}


def test_a_short_layer_covers_the_last_tokens_not_the_first(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1], [2], [3], [4]]), rec(0, 9, [[7], [8]])])
    rows = read_trace(p)
    assert 9 not in rows[0].routing and 9 not in rows[1].routing
    assert rows[2].routing[9] == (7,)
    assert rows[3].routing[9] == (8,)


def test_a_partly_routed_token_reports_a_missing_layer_not_an_empty_one(tmp_path):
    """Those are different claims, and a predictor scored on them differs."""
    p = write(tmp_path, [rec(0, 1, [[1], [2]]), rec(0, 2, [[3]])])
    rows = read_trace(p)
    assert 2 not in rows[0].routing
    assert rows[0].routing.get(2) is not ()


# -- refusing what is actually broken -----------------------------------


def test_a_layer_missing_from_a_whole_decode_is_refused(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1]]), rec(0, 2, [[2]]), rec(1, 1, [[3]])])
    with pytest.raises(TraceError, match="missing layers"):
        read_trace(p)


def test_a_row_count_disagreeing_with_its_own_header_is_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"decode": 0, "layer": 1, "n_tokens": 5, "n_used": 1,
                             "experts": [[1], [2]]}) + "\n")
    with pytest.raises(TraceError, match="claims 5 tokens"):
        read_trace(p)


def test_a_repeated_layer_within_a_decode_is_refused(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1]]), rec(0, 1, [[2]])])
    with pytest.raises(TraceError, match="repeats layer"):
        read_trace(p)


def test_a_line_that_is_not_a_record_names_its_line_number(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"decode": 0, "layer": 1, "n_tokens": 1,
                             "n_used": 1, "experts": [[1]]}) + "\nnot json\n")
    with pytest.raises(TraceError, match=":2:"):
        read_trace(p)


def test_an_empty_trace_is_refused_rather_than_scored_as_zero(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n\n")
    with pytest.raises(TraceError, match="no routing records"):
        read_trace(p)


def test_blank_lines_between_records_are_ignored(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(rec(0, 1, [[1]])) + "\n\n" + json.dumps(rec(1, 1, [[2]])) + "\n")
    assert len(read_trace(p)) == 2


# -- describing one -----------------------------------------------------


def test_describe_counts_what_a_reader_needs_to_know(tmp_path):
    p = write(tmp_path, [rec(0, 1, [[1, 2], [2, 3]]), rec(0, 2, [[4, 5]]),
                         rec(1, 1, [[1, 9]]), rec(1, 2, [[4, 5]])])
    info = describe(p)
    assert info.decodes == 2
    assert info.prompt_tokens == 2
    assert info.generated_tokens == 1
    assert info.layers == (1, 2)
    assert info.n_used == 2
    assert info.experts_seen == 6           # 1,2,3,4,5,9
    assert info.partial_tokens == 1         # the first prompt token lacks layer 2
    assert info.tokens == 3

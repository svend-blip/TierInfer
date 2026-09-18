"""Tests for the telemetry schema.

What matters is that a record written behind one runtime can be read beside
one written behind another, and that nothing is silently absent — a missing
column reads as a zero later, and a zero that was never measured is the most
expensive kind of wrong number.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.telemetry import (  # noqa: E402
    SCHEMA_VERSION, Event, Snapshot, Telemetry, TelemetryError, collect,
    deltas, read, STANDARD_FIELDS,
)


class Counters:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def tel(tmp_path):
    with Telemetry(tmp_path / "t.jsonl", run="testrun") as t:
        yield t


# -- records ------------------------------------------------------------


def test_every_record_carries_the_schema_version_and_the_run(tel):
    tel.open_run()
    tel.event("thing.happened", n=1)
    recs = read(tel.path)
    assert all(r["v"] == SCHEMA_VERSION for r in recs)
    assert all(r["run"] == "testrun" for r in recs)


def test_an_event_carries_its_own_fields(tel):
    tel.open_run()
    tel.event("policy.depth", depth=12, reason="stalls")
    rec = read(tel.path)[-1]
    assert rec["type"] == "event" and rec["kind"] == "policy.depth"
    assert rec["depth"] == 12 and rec["reason"] == "stalls"


def test_a_snapshot_namespaces_every_counter(tel):
    tel.open_run()
    tel.snapshot({"cache": Counters(hits=3, misses=1, insertions=4, evictions=0,
                                    bytes_admitted=99, bytes_evicted=0)})
    rec = read(tel.path)[-1]
    assert rec["type"] == "snapshot"
    assert rec["values"]["cache.hits"] == 3
    assert "hits" not in rec["values"], "counters must be namespaced"


def test_nested_values_are_refused_rather_than_flattened(tel):
    tel.open_run()
    with pytest.raises(TelemetryError, match="scalar"):
        tel.event("bad", payload={"a": 1})


def test_a_counter_that_is_a_method_is_called(tel):
    class WithProperty:
        bytes_read = 10
        operations = 2

        def seconds(self):
            return 0.5

    got = collect("storage", WithProperty(), STANDARD_FIELDS["storage"])
    assert got["storage.seconds"] == 0.5


def test_a_missing_counter_is_an_error_not_a_gap():
    with pytest.raises(TelemetryError, match="no 'misses'"):
        collect("cache", Counters(hits=1), ("hits", "misses"))


def test_a_namespace_with_no_field_list_is_refused(tel):
    tel.open_run()
    with pytest.raises(TelemetryError, match="STANDARD_FIELDS"):
        tel.snapshot({"invented": Counters(x=1)})


def test_a_source_that_is_none_is_skipped_not_an_error(tel):
    tel.open_run()
    s = tel.snapshot({"cache": None, "sim": None})
    assert s.values == {}


# -- the run's frame ----------------------------------------------------


def test_the_opening_event_says_where_it_ran(tel):
    tel.open_run(model="glm4moe", context=16384)
    rec = read(tel.path)[0]
    assert rec["kind"] == "run.open"
    for key in ("host", "platform", "python", "pid", "model", "context"):
        assert key in rec


def test_a_file_that_does_not_open_with_run_open_is_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    with Telemetry(p, run="x") as t:
        t.event("something", n=1)
    with pytest.raises(TelemetryError, match="run.open"):
        read(p)


def test_an_empty_file_is_refused_rather_than_read_as_a_quiet_run(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("")
    with pytest.raises(TelemetryError, match="no records"):
        read(p)


def test_a_record_from_a_future_schema_is_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"v": SCHEMA_VERSION + 1, "type": "event",
                             "kind": "run.open", "at": 0, "run": "x"}) + "\n")
    with pytest.raises(TelemetryError, match="schema version"):
        read(p)


def test_a_line_that_is_not_json_names_its_line_number(tmp_path):
    p = tmp_path / "t.jsonl"
    with Telemetry(p, run="x") as t:
        t.open_run()
    with open(p, "a") as fh:
        fh.write("not json\n")
    with pytest.raises(TelemetryError, match=":2:"):
        read(p)


def test_records_land_as_they_are_written_not_at_close(tmp_path):
    """A reader arriving mid-run gets everything that has happened."""
    t = Telemetry(tmp_path / "t.jsonl", run="x")
    t.open_run()
    t.event("mid", n=1)
    assert len(read(t.path)) == 2, "records were still in a buffer"
    t.close()


def test_telemetry_without_a_path_still_counts_what_it_would_write():
    t = Telemetry(None, run="x")
    t.open_run()
    t.snapshot({"cache": Counters(hits=1, misses=0, insertions=1, evictions=0,
                                  bytes_admitted=1, bytes_evicted=0)})
    assert t.records == 2


# -- what a reader does with it -----------------------------------------


def test_deltas_difference_consecutive_snapshots(tel):
    tel.open_run()
    for hits in (0, 5, 12):
        tel.snapshot({"cache": Counters(hits=hits, misses=0, insertions=0,
                                        evictions=0, bytes_admitted=0,
                                        bytes_evicted=0)})
    got = list(deltas(read(tel.path)))
    assert [d["cache.hits"] for d in got] == [5, 7]
    assert all(d["seconds"] >= 0 for d in got)


def test_a_key_absent_from_one_end_is_skipped_rather_than_assumed_zero(tel):
    tel.open_run()
    tel.snapshot({"cache": Counters(hits=1, misses=0, insertions=0, evictions=0,
                                    bytes_admitted=0, bytes_evicted=0)})
    tel.snapshot({"sim": Counters(tokens=1, lookups=1, vram_hits=1, ram_hits=0,
                                     nvme_reads=0, prefetched=0, prefetch_used=0,
                                     stalls=0, seconds=0.0, prefetch_seconds=0.0,
                                     depth_changes=0)})
    got = list(deltas(read(tel.path)))
    assert "cache.hits" not in got[0]
    assert "policy.tokens" not in got[0]


def test_events_between_snapshots_do_not_disturb_the_deltas(tel):
    tel.open_run()
    tel.snapshot({"cache": Counters(hits=0, misses=0, insertions=0, evictions=0,
                                    bytes_admitted=0, bytes_evicted=0)})
    tel.event("noise", n=1)
    tel.snapshot({"cache": Counters(hits=4, misses=0, insertions=0, evictions=0,
                                    bytes_admitted=0, bytes_evicted=0)})
    got = list(deltas(read(tel.path)))
    assert len(got) == 1 and got[0]["cache.hits"] == 4


def test_a_single_snapshot_yields_no_deltas(tel):
    tel.open_run()
    tel.snapshot({"cache": Counters(hits=1, misses=0, insertions=0, evictions=0,
                                    bytes_admitted=0, bytes_evicted=0)})
    assert list(deltas(read(tel.path))) == []


# -- the schema is shared, which is the whole point ---------------------


def test_the_real_components_expose_the_fields_the_schema_names():
    """A schema naming fields nothing has is a schema nobody can fill."""
    from tierinfer.cache import CacheStats
    from tierinfer.policy import PolicyStats
    from tierinfer.prefetch import PrefetchStats
    from tierinfer.storage import StorageStats
    from tierinfer.stream import StreamStats
    from tierinfer.vram import VramStats

    for namespace, stats in (("cache", CacheStats()), ("sim", PolicyStats()),
                             ("prefetch", PrefetchStats()), ("storage", StorageStats()),
                             ("stream", StreamStats()), ("vram", VramStats())):
        collect(namespace, stats, STANDARD_FIELDS[namespace])


def test_two_runs_behind_different_runtimes_share_a_shape(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, runtime in ((a, "llama.cpp"), (b, "simulation")):
        with Telemetry(path, run=runtime) as t:
            t.open_run(runtime=runtime)
            t.snapshot({"sim": Counters(tokens=1, lookups=360, vram_hits=300,
                                           ram_hits=50, nvme_reads=10, prefetched=8,
                                           prefetch_used=5, stalls=10, seconds=0.07,
                                           prefetch_seconds=0.01, depth_changes=0)})
    ka = set(read(a)[-1]["values"])
    kb = set(read(b)[-1]["values"])
    assert ka == kb and ka, "the same components must produce the same keys"

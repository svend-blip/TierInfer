"""Tests for the FreeToken adapter.

FreeToken owns its loading path and tiers experts itself, so what is tested
here is the two things the adapter actually does: hand it a budget TierInfer
derived, and take its numbers back in the shared shape.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.adapters.freetoken import (  # noqa: E402
    RUNTIME_FIELDS, FreeTokenClient, FreeTokenError, FreeTokenUnreachable,
    describe_launch, launch_arguments, normalise,
)
from tierinfer.autoconfig import Configuration, Host  # noqa: E402
from tierinfer.telemetry import Telemetry, read  # noqa: E402
from tierinfer.vram import GB, MB  # noqa: E402


def host():
    return Host(ram_total=192 * GB, ram_available=176 * GB, vram_total=32 * GB,
                vram_free=31 * GB, cpu_count=32, platform="test")


def config(usable=True, vram=2000, ram=5000, context=8192):
    return Configuration(
        model="fake.gguf", context_length=context, model_bytes=56 * GB,
        floor_bytes=4 * GB, expert_bytes=10 * MB, vram_experts=vram,
        ram_experts=ram, stream_workers=8, prefetch_depth=8, budget=None,
        host=host(), problems=[] if usable else ["context does not fit"])


# FreeToken's own /v1/stats shape, from python/freetoken/server/stats.py.
STATS = {
    "instance_id": "ft-1",
    "uptime_s": 412.5,
    "kv": {"used_pages": 120, "total_pages": 4096, "page_size": 16},
    "vram_bytes": 22_000_000_000,
    "throughput": {"decode_tps": 41.2, "prefill_tps": 780.5},
    "requests": {"active": 1, "completed": 37},
    "model": {"name": "glm4moe"},
}


# -- configuration in ---------------------------------------------------


def test_the_flags_carry_the_derived_cache_size_and_context():
    args = launch_arguments(config(vram=2000))
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--moe-cache-size"] == f"{2000 * 10 * MB // MB}M"
    assert pairs["--kv-reserve-tokens"] == "8192"
    assert pairs["--moe-cache-policy"] == "lru"


def test_a_host_without_a_card_falls_back_to_the_ram_budget():
    args = launch_arguments(config(vram=0, ram=5000))
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--moe-cache-size"] == f"{5000 * 10 * MB // MB}M"


def test_an_unusable_configuration_is_refused_before_a_server_starts():
    """FreeToken would discover this after loading 56 GB."""
    with pytest.raises(FreeTokenError, match="after loading the model"):
        launch_arguments(config(usable=False))


def test_a_configuration_with_no_room_for_a_cache_is_refused():
    with pytest.raises(FreeTokenError, match="no room"):
        launch_arguments(config(vram=0, ram=0))


def test_the_memory_ratio_is_optional_and_bounded():
    assert "--memory-ratio" not in launch_arguments(config())
    assert "--memory-ratio" in launch_arguments(config(), memory_ratio=0.8)
    for bad in (0.0, -1.0, 1.5):
        with pytest.raises(FreeTokenError, match="memory ratio"):
            launch_arguments(config(), memory_ratio=bad)


def test_the_launch_description_says_what_each_flag_came_from():
    text = describe_launch(config())
    assert "--moe-cache-size" in text
    assert "derived from:" in text
    assert "context asked for" in text


# -- telemetry out ------------------------------------------------------


def test_the_stats_document_becomes_the_shared_namespace():
    got = normalise(STATS)
    assert set(got) == set(RUNTIME_FIELDS)
    assert got["runtime.decode_tps"] == 41.2
    assert got["runtime.kv_used_pages"] == 120
    assert got["runtime.vram_bytes"] == 22_000_000_000


def test_a_counter_the_runtime_does_not_report_is_none_not_zero():
    """A runtime that does not report a counter has not reported zero of it,
    and the difference matters the moment two runtimes are compared."""
    got = normalise({"uptime_s": 1.0})
    assert got["runtime.decode_tps"] is None
    assert got["runtime.kv_used_pages"] is None
    assert got["runtime.uptime_s"] == 1.0


def test_an_empty_document_normalises_rather_than_raising():
    got = normalise({})
    assert set(got) == set(RUNTIME_FIELDS)
    assert all(v is None for v in got.values())


def test_something_that_is_not_a_document_is_refused():
    with pytest.raises(FreeTokenError, match="mapping"):
        normalise([1, 2, 3])


def test_normalised_values_go_into_a_snapshot_unchanged(tmp_path):
    with Telemetry(tmp_path / "t.jsonl", run="ft") as t:
        t.open_run(runtime="freetoken")
        t.snapshot(values=normalise(STATS))
    rec = read(tmp_path / "t.jsonl")[-1]
    assert rec["values"]["runtime.decode_tps"] == 41.2


def test_a_bare_counter_name_is_refused_by_the_telemetry(tmp_path):
    with Telemetry(tmp_path / "t.jsonl", run="ft") as t:
        t.open_run()
        with pytest.raises(Exception, match="namespaced"):
            t.snapshot(values={"decode_tps": 1.0})


# -- the client ---------------------------------------------------------


def test_an_unreachable_server_is_a_fact_not_a_crash():
    c = FreeTokenClient("http://127.0.0.1:9", timeout=0.5)
    assert c.is_up() is False
    with pytest.raises(FreeTokenUnreachable):
        c.stats()


def test_the_client_reads_what_the_server_returns(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._p = json.dumps(payload).encode()

        def read(self):
            return self._p

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return FakeResponse(STATS)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = FreeTokenClient("http://host:8080/")
    assert c.snapshot_values()["runtime.decode_tps"] == 41.2
    assert seen["url"] == "http://host:8080/v1/stats"


def test_the_adapter_never_touches_a_weight_or_a_device():
    """Goal 6 measured what happens to an adapter that tries to manage
    residency in a runtime that owns its own: nothing, slowly."""
    import tierinfer.adapters.freetoken as mod
    source = open(mod.__file__).read()
    for forbidden in ("posix_fadvise", "cudaMalloc", "mmap", "pread"):
        assert forbidden not in source, f"the adapter reaches for {forbidden}"

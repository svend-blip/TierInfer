"""Tests for deriving a configuration from the host and the model.

The point of these is the refusals. A configuration that returns plausible
numbers for a setup that cannot work costs a run to discover, so the cases
that must fail get more attention here than the case that must succeed.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.autoconfig import (  # noqa: E402
    DEFAULT_RAM_SHARE, Configuration, Host, configure,
)
from tierinfer.vram import GB, MB  # noqa: E402

GLM_META = {
    "general.architecture": "glm4moe",
    "glm4moe.attention.head_count_kv": 8,
    "glm4moe.attention.key_length": 128,
    "glm4moe.attention.value_length": 128,
}


class FakeGGUF:
    path = "/models/fake.gguf"
    metadata = GLM_META


class FakeIndex:
    """A model's shape, without a 56 GB file."""

    def __init__(self, floor=4 * GB, routed=52 * GB, expert=10 * MB,
                 layers=46, used=8, blocks=47):
        self.gguf = FakeGGUF()
        self.block_count = blocks
        self.expert_used_count = used
        self.moe_layers = list(range(layers))
        self._floor, self._routed, self._expert = floor, routed, expert

    def always_resident_nbytes(self):
        return self._floor

    def routed_nbytes(self):
        return self._routed

    def expert_nbytes(self):
        return self._expert


def host(ram=192 * GB, avail=176 * GB, vram=32 * GB, vram_free=31 * GB, cpus=32):
    return Host(ram_total=ram, ram_available=avail, vram_total=vram,
                vram_free=vram_free, cpu_count=cpus, platform="test")


# -- the ordinary case --------------------------------------------------


def test_a_workstation_gets_a_usable_configuration():
    c = configure(FakeIndex(), context_length=8192, host=host())
    assert c.usable and not c.problems
    assert c.vram_experts > 0 and c.ram_experts > 0
    assert c.stream_workers == 8


def test_every_decision_is_stated_rather_than_implied():
    c = configure(FakeIndex(), context_length=8192, host=host())
    text = c.explain()
    assert "decisions:" in text
    assert any("reserve" in d for d in c.decisions)
    assert any("RAM share" in d for d in c.decisions)


def test_the_configuration_round_trips_as_data():
    c = configure(FakeIndex(), context_length=8192, host=host())
    d = c.to_dict()
    assert d["usable"] is True
    assert d["vram_experts"] == c.vram_experts
    assert isinstance(d["decisions"], list)


def test_more_context_leaves_fewer_vram_experts():
    counts = [configure(FakeIndex(), context_length=c, host=host()).vram_experts
              for c in (4096, 16384, 65536)]
    assert counts == sorted(counts, reverse=True)


def test_the_measured_reserve_is_used_rather_than_a_default():
    busy = configure(FakeIndex(), context_length=8192,
                     host=host(vram_free=20 * GB))
    idle = configure(FakeIndex(), context_length=8192,
                     host=host(vram_free=31 * GB))
    assert busy.vram_experts < idle.vram_experts


# -- the refusals -------------------------------------------------------


def test_a_context_whose_kv_cache_does_not_fit_is_refused():
    c = configure(FakeIndex(), context_length=131072, host=host(vram=8 * GB,
                                                               vram_free=8 * GB))
    assert not c.usable
    assert any("does not fit" in p for p in c.problems)


def test_a_vram_budget_below_one_tokens_working_set_is_refused():
    """Goal 9 measured this: below one token's experts the hit rate is zero,
    not low, so a configuration that lands there is not a slow one."""
    c = configure(FakeIndex(), context_length=131072, host=host())
    assert not c.usable
    assert any("one token routes to" in p for p in c.problems)


def test_a_host_that_cannot_hold_a_single_expert_is_refused():
    tiny = Host(ram_total=GB, ram_available=GB, vram_total=None, vram_free=None,
                cpu_count=2, platform="test")
    c = configure(FakeIndex(floor=2 * GB), context_length=1024, host=tiny)
    assert not c.usable
    assert any("single expert" in p for p in c.problems)


def test_problems_are_reported_together_rather_than_one_at_a_time():
    c = configure(FakeIndex(), context_length=131072, host=host(vram=6 * GB,
                                                               vram_free=6 * GB))
    assert not c.usable
    assert "problems:" in c.explain()


# -- a host without a card ----------------------------------------------


def test_a_host_with_no_gpu_is_configured_for_ram_alone():
    cpu_only = Host(ram_total=64 * GB, ram_available=48 * GB, vram_total=None,
                    vram_free=None, cpu_count=16, platform="test")
    c = configure(FakeIndex(), context_length=8192, host=cpu_only)
    assert c.usable
    assert c.vram_experts == 0 and c.budget is None
    assert any("no GPU" in d for d in c.decisions)
    assert "no GPU" in c.explain()


def test_a_missing_gpu_is_reported_absent_not_as_zero():
    h = Host(ram_total=GB, ram_available=GB, vram_total=None, vram_free=None,
             cpu_count=1, platform="test")
    assert h.has_gpu is False
    assert h.vram_total is None, "absent must not be spelled as zero"


# -- the shares ---------------------------------------------------------


def test_ram_is_shared_rather_than_claimed_whole():
    c = configure(FakeIndex(), context_length=8192, host=host(),
                  ram_share=DEFAULT_RAM_SHARE)
    assert c.ram_bytes < 176 * GB
    assert DEFAULT_RAM_SHARE < 1.0, "planning around all of RAM leaves nothing"


def test_a_larger_share_plans_for_more_experts():
    small = configure(FakeIndex(), context_length=8192, host=host(), ram_share=0.3)
    large = configure(FakeIndex(), context_length=8192, host=host(), ram_share=0.8)
    assert large.ram_experts > small.ram_experts


def test_workers_are_bounded_by_the_cpus_present():
    c = configure(FakeIndex(), context_length=8192, host=host(cpus=4))
    assert c.stream_workers == 2


def test_an_explicit_worker_count_is_taken_without_a_decision_note():
    c = configure(FakeIndex(), context_length=8192, host=host(), stream_workers=3)
    assert c.stream_workers == 3
    assert not any("stream workers" in d for d in c.decisions)


def test_resident_share_says_how_much_of_the_model_is_in_memory():
    c = configure(FakeIndex(routed=52 * GB), context_length=8192,
                  host=host(avail=20 * GB))
    assert 0.0 < c.resident_share < 1.0
    assert "resident share" in c.explain()


def test_a_host_big_enough_holds_everything():
    c = configure(FakeIndex(), context_length=8192, host=host(avail=400 * GB))
    assert c.resident_share == 1.0


# -- measuring the real host -------------------------------------------


def test_measuring_this_host_reports_what_it_has():
    h = Host.measure()
    assert h.cpu_count >= 1
    assert h.ram_total > 0, "/proc/meminfo should have been readable"
    assert h.ram_available > 0
    if h.has_gpu:
        assert h.vram_total > h.vram_free >= 0

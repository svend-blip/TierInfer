"""Tests for the FlowRunner capability.

The split this enforces is the whole design: a flow declares what the *work*
needs, a host decides what it can give. A declaration that could name a VRAM
size would run on one machine and fail on the next, so naming one is refused.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.adapters.flowrunner import (  # noqa: E402
    CAPABILITY_NAME, CAPABILITY_VERSION, Capability, CapabilityError,
    resolve, telemetry_values,
)
from tierinfer.autoconfig import Host  # noqa: E402
from tierinfer.telemetry import Telemetry, read  # noqa: E402
from tierinfer.vram import GB, MB  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_autoconfig import FakeIndex, host  # noqa: E402


def doc(**kw):
    base = {"version": CAPABILITY_VERSION, "capability": CAPABILITY_NAME,
            "model": "glm.gguf"}
    base.update(kw)
    return base


# -- reading a declaration ---------------------------------------------


def test_a_minimal_declaration_is_enough():
    c = Capability.from_dict(doc())
    assert c.model == "glm.gguf"
    assert c.context_length == 8192


def test_a_declaration_round_trips():
    c = Capability.from_dict(doc(context_length=16384, min_resident_share=0.9,
                                 notes="needs speed"))
    assert Capability.from_dict(c.to_dict()) == c


def test_a_declaration_loads_from_a_file(tmp_path):
    p = tmp_path / "cap.json"
    p.write_text(json.dumps(doc(context_length=4096)))
    assert Capability.load(p).context_length == 4096


def test_a_file_that_is_not_json_names_itself(tmp_path):
    p = tmp_path / "cap.json"
    p.write_text("{not json")
    with pytest.raises(CapabilityError, match="not JSON"):
        Capability.load(p)


def test_a_missing_file_is_a_capability_error_not_an_oserror(tmp_path):
    with pytest.raises(CapabilityError):
        Capability.load(tmp_path / "absent.json")


# -- what a declaration may not say ------------------------------------


@pytest.mark.parametrize("field", ["vram_bytes", "vram_gb", "cache_size",
                                   "moe_cache_size"])
def test_a_flow_may_not_name_a_host_property(field):
    """A flow declaring 22 GB of cache runs here and fails on the next
    machine. It declares a context; the host derives the rest."""
    with pytest.raises(CapabilityError, match="property of the host"):
        Capability.from_dict(doc(**{field: 22 * GB}))


def test_an_unknown_field_is_refused_rather_than_ignored():
    with pytest.raises(CapabilityError, match="unknown field"):
        Capability.from_dict(doc(contxt_length=4096))


def test_a_missing_required_field_says_which():
    d = doc()
    del d["model"]
    with pytest.raises(CapabilityError, match="model"):
        Capability.from_dict(d)


def test_a_declaration_from_another_version_is_refused():
    with pytest.raises(CapabilityError, match="version"):
        Capability.from_dict(doc(version=CAPABILITY_VERSION + 1))


def test_a_declaration_of_another_capability_is_refused():
    with pytest.raises(CapabilityError, match="capability"):
        Capability.from_dict(doc(capability="something.else"))


@pytest.mark.parametrize("bad", [0, -1])
def test_a_context_that_is_not_positive_is_refused(bad):
    with pytest.raises(CapabilityError, match="context_length"):
        Capability.from_dict(doc(context_length=bad))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_resident_share_outside_zero_to_one_is_refused(bad):
    with pytest.raises(CapabilityError, match="min_resident_share"):
        Capability.from_dict(doc(min_resident_share=bad))


def test_something_that_is_not_an_object_is_refused():
    with pytest.raises(CapabilityError, match="object"):
        Capability.from_dict([1, 2, 3])


# -- resolving against a host ------------------------------------------


def test_a_workstation_resolves_a_reasonable_declaration():
    r = resolve(Capability.from_dict(doc(context_length=8192)), FakeIndex(),
                host=host())
    assert r.available
    assert r.configuration.vram_experts > 0
    assert "asked for" in r.explain()


def test_a_context_the_host_cannot_serve_is_refused_with_the_reason():
    r = resolve(Capability.from_dict(doc(context_length=131072)), FakeIndex(),
                host=host())
    assert not r.available
    assert r.refusals
    assert "not available here:" in r.explain()


def test_a_step_that_needs_speed_can_say_so_and_be_refused():
    """A flow that only makes sense at speed should not run slowly and be
    believed."""
    small = Host(ram_total=16 * GB, ram_available=8 * GB, vram_total=None,
                 vram_free=None, cpu_count=4, platform="test")
    cap = Capability.from_dict(doc(min_resident_share=0.9))
    r = resolve(cap, FakeIndex(), host=small)
    assert not r.available
    assert any("below the 90%" in x for x in r.refusals)


def test_the_same_declaration_is_available_where_the_host_is_big_enough():
    cap = Capability.from_dict(doc(min_resident_share=0.9))
    assert resolve(cap, FakeIndex(), host=host()).available


def test_a_declared_worker_count_reaches_the_configuration():
    cap = Capability.from_dict(doc(stream_workers=3))
    assert resolve(cap, FakeIndex(), host=host()).configuration.stream_workers == 3


def test_a_declared_ram_share_reaches_the_configuration():
    a = resolve(Capability.from_dict(doc(ram_share=0.3)), FakeIndex(), host=host())
    b = resolve(Capability.from_dict(doc(ram_share=0.8)), FakeIndex(), host=host())
    assert b.configuration.ram_experts > a.configuration.ram_experts


def test_the_declared_prefetch_depth_is_a_starting_point_that_is_carried():
    cap = Capability.from_dict(doc(prefetch_depth=16))
    assert resolve(cap, FakeIndex(), host=host()).configuration.prefetch_depth == 16


# -- telemetry out ------------------------------------------------------


def test_a_resolved_step_reports_itself_in_the_shared_schema(tmp_path):
    r = resolve(Capability.from_dict(doc()), FakeIndex(), host=host())
    values = telemetry_values(r.configuration)
    assert all("." in k for k in values), "values must be namespaced"
    with Telemetry(tmp_path / "t.jsonl", run="fr") as t:
        t.open_run(runtime="flowrunner")
        t.snapshot(values=values)
    rec = read(tmp_path / "t.jsonl")[-1]
    assert rec["values"]["capability.usable"] is True
    assert rec["values"]["capability.context_length"] == 8192


def test_the_reported_values_are_all_scalar():
    values = telemetry_values(
        resolve(Capability.from_dict(doc()), FakeIndex(), host=host()).configuration)
    for v in values.values():
        assert isinstance(v, (int, float, str, bool)) or v is None

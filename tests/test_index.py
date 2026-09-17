"""Tests for expert addressing: the arithmetic that turns a name into a byte range."""

from __future__ import annotations

import pytest

from tierinfer.gguf import GGUFError
from tierinfer.index import ModelIndex
from tierinfer.gguf import read_gguf

from test_gguf import IQ4_XS, F32, write_gguf

EXPERTS = 8


def _model(tmp_path, experts=EXPERTS, layers=2):
    tensors = [("token_embd.weight", (256, 8), F32)]
    for l in range(layers):
        tensors += [
            (f"blk.{l}.attn_q.weight", (256, 8), IQ4_XS),
            (f"blk.{l}.attn_norm.weight", (256,), F32),
            (f"blk.{l}.ffn_gate_inp.weight", (256, experts), F32),
            (f"blk.{l}.ffn_gate_shexp.weight", (256, 4), IQ4_XS),
            (f"blk.{l}.ffn_gate_exps.weight", (256, 4, experts), IQ4_XS),
            (f"blk.{l}.ffn_up_exps.weight", (256, 4, experts), IQ4_XS),
            (f"blk.{l}.ffn_down_exps.weight", (256, 4, experts), IQ4_XS),
        ]
    p = write_gguf(tmp_path / "m.gguf", tensors, {
        "general.architecture": "glm4moe",
        "glm4moe.expert_count": experts,
        "glm4moe.expert_used_count": 2,
        "glm4moe.block_count": layers,
    })
    return ModelIndex(read_gguf(p))


def test_an_expert_is_a_slice_of_each_fused_projection(tmp_path):
    ix = _model(tmp_path)
    ref = ix.expert(1, 3)
    assert len(ref.ranges) == 3
    fused = ix.gguf.tensor("blk.1.ffn_gate_exps.weight")
    slab = fused.nbytes // EXPERTS
    gate = [r for r in ref.ranges if "gate" in r.name][0]
    assert gate.file_offset == fused.file_offset + 3 * slab
    assert gate.nbytes == slab


def test_consecutive_experts_are_adjacent_and_cover_the_tensor(tmp_path):
    ix = _model(tmp_path)
    fused = ix.gguf.tensor("blk.0.ffn_down_exps.weight")
    ranges = [[r for r in ix.expert(0, e).ranges if "down" in r.name][0] for e in range(EXPERTS)]
    assert ranges[0].file_offset == fused.file_offset
    for a, b in zip(ranges, ranges[1:]):
        assert a.end == b.file_offset
    assert ranges[-1].end == fused.file_offset + fused.nbytes


def test_an_out_of_range_expert_refuses(tmp_path):
    ix = _model(tmp_path)
    with pytest.raises(GGUFError):
        ix.expert(0, EXPERTS)


def test_a_layer_without_routed_experts_refuses(tmp_path):
    ix = _model(tmp_path)
    with pytest.raises(GGUFError):
        ix.expert(99, 0)


def test_the_working_set_is_the_floor_plus_what_a_token_routes_to(tmp_path):
    ix = _model(tmp_path)
    per_expert = ix.expert_nbytes()
    expected = ix.always_resident_nbytes() + len(ix.moe_layers) * ix.expert_used_count * per_expert
    assert ix.working_set_nbytes() == expected
    assert ix.working_set_nbytes() < ix.always_resident_nbytes() + ix.routed_nbytes()


def test_the_resident_floor_excludes_routed_experts_and_includes_the_shared_one(tmp_path):
    ix = _model(tmp_path)
    layer = ix.layers[0]
    assert layer.is_moe
    assert any("shexp" in t.name for t in layer.shared_expert)
    assert all("_exps" not in t.name for group in
               (layer.attention, layer.norms, layer.router, layer.shared_expert)
               for t in group)

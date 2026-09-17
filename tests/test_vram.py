"""Tests for the VRAM budget and residency.

The budget arithmetic is tested without a card, because it is arithmetic and
because a host without CUDA should still be able to answer "would this fit".
The residency tests need a device and skip where there is none, rather than
asserting something the host cannot answer.
"""

from __future__ import annotations

import ctypes
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.vram import (  # noqa: E402
    GB, MB, CudaError, CudaRuntime, CudaUnavailable, DeviceMemory, VramBudget,
    VramResidency, VramStats,
)

# GLM-4.5-Air-Derestricted, read from the real file.
GLM = {
    "general.architecture": "glm4moe",
    "glm4moe.attention.head_count": 96,
    "glm4moe.attention.head_count_kv": 8,
    "glm4moe.attention.key_length": 128,
    "glm4moe.attention.value_length": 128,
}


def cuda_or_skip():
    try:
        return CudaRuntime()
    except CudaUnavailable:
        pytest.skip("no CUDA runtime on this host")


# -- the KV arithmetic, checked against llama.cpp's own allocation ------


@pytest.mark.parametrize("context,mib", [(2048, 376), (8192, 1504), (16384, 3008)])
def test_kv_matches_what_llama_cpp_allocates(context, mib):
    """Measured from llama.cpp b9888 on this model at these three contexts.
    The KV term is the largest in the budget; if it drifts, everything
    downstream is wrong and nothing else would notice."""
    per_token = VramBudget.kv_bytes_per_token(GLM, layers=47)
    assert per_token * context == mib * MB


def test_grouped_query_attention_is_what_makes_this_small():
    """96 query heads against 8 key/value heads. Using head_count here would
    overstate the KV cache twelvefold."""
    per_token = VramBudget.kv_bytes_per_token(GLM, layers=47)
    assert per_token == 47 * 8 * (128 + 128) * 2


def test_a_narrower_kv_type_costs_proportionally_less():
    f16 = VramBudget.kv_bytes_per_token(GLM, layers=47, kv_bits=16)
    q8 = VramBudget.kv_bytes_per_token(GLM, layers=47, kv_bits=8)
    assert q8 * 2 == f16


def test_a_model_without_the_metadata_is_refused_not_guessed():
    with pytest.raises(CudaError, match="head_count_kv"):
        VramBudget.kv_bytes_per_token({"general.architecture": "mystery"}, layers=47)


# -- the budget ---------------------------------------------------------


def budget(context, total=32 * GB, reserve=1 * GB):
    return VramBudget.from_model(GLM, total_bytes=total, context_length=context,
                                 layers=47, reserve_bytes=reserve)


def test_every_claim_is_subtracted_and_the_remainder_is_for_weights():
    b = budget(16384)
    assert b.weights == b.total - b.reserve - b.kv_cache - b.runtime_overhead
    assert b.fits


def test_a_context_that_does_not_fit_says_so_rather_than_going_negative_quietly():
    b = budget(131072, total=8 * GB)
    assert not b.fits
    assert b.weights < 0, "a negative remainder is the answer, not an error"


def test_experts_never_reports_a_negative_count():
    b = budget(131072, total=8 * GB)
    assert b.experts(10 * MB) == 0
    assert b.experts(10 * MB, floor_bytes=100 * GB) == 0


def test_the_floor_comes_out_before_the_experts_are_counted():
    b = budget(4096)
    with_floor = b.experts(10 * MB, floor_bytes=4 * GB)
    without = b.experts(10 * MB)
    assert without - with_floor == pytest.approx(4 * GB // (10 * MB), rel=0.01)


def test_more_context_leaves_fewer_experts():
    counts = [budget(c).experts(10 * MB, 4 * GB) for c in (4096, 16384, 65536)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_an_expert_size_of_zero_does_not_divide_by_zero():
    assert budget(4096).experts(0) == 0


def test_explain_names_every_term_it_subtracts():
    text = budget(16384).explain()
    for term in ("total", "reserve", "KV cache", "runtime overhead", "left for weights"):
        assert term in text


def test_the_runtime_overhead_default_sits_above_what_was_measured():
    """llama.cpp reported 330, 328 and 320 MiB; the default must cover them."""
    assert VramBudget.MEASURED_RUNTIME_OVERHEAD >= 330 * MB


# -- statistics ---------------------------------------------------------


def test_stats_of_an_untouched_residency_do_not_divide_by_zero():
    s = VramStats()
    assert s.hit_rate == 0.0
    assert s.bytes_per_second == 0.0
    assert s.mean_transfer_seconds == 0.0


def test_device_memory_reports_what_is_used():
    m = DeviceMemory(free=4 * GB, total=10 * GB)
    assert m.used == 6 * GB


# -- residency, on a real device ---------------------------------------


def test_a_pool_takes_exactly_what_it_asked_for_and_gives_it_all_back():
    rt = cuda_or_skip()
    before = rt.memory_info().free
    res = VramResidency(rt, 4 * MB, 16)
    during = rt.memory_info().free
    assert res.device_bytes == 64 * MB
    assert before - during >= 64 * MB
    res.close()
    assert rt.memory_info().free >= before - 1 * MB, "the pool leaked device memory"


def test_a_resident_expert_is_found_and_a_missing_one_is_not():
    rt = cuda_or_skip()
    host = rt.host_alloc(MB)
    with VramResidency(rt, MB, 4) as res:
        assert res.lookup(("a", 1)) is None
        ptr = res.admit(("a", 1), host, MB)
        assert res.lookup(("a", 1)) == ptr
        assert ("a", 1) in res and len(res) == 1
    rt.host_free(host)


def test_admitting_the_same_expert_twice_does_not_transfer_twice():
    rt = cuda_or_skip()
    host = rt.host_alloc(MB)
    with VramResidency(rt, MB, 4) as res:
        first = res.admit(("a", 1), host, MB)
        assert res.admit(("a", 1), host, MB) == first
        assert res.stats.transfers == 1
    rt.host_free(host)


def test_a_full_pool_evicts_rather_than_failing():
    rt = cuda_or_skip()
    host = rt.host_alloc(MB)
    with VramResidency(rt, MB, 2) as res:
        for i in range(5):
            res.admit(("a", i), host, MB)
        assert len(res) == 2, "the pool grew past its slots"
        assert res.stats.evictions == 3
    rt.host_free(host)


def test_an_expert_larger_than_a_slot_is_refused_not_truncated():
    rt = cuda_or_skip()
    host = rt.host_alloc(2 * MB)
    with VramResidency(rt, MB, 2) as res:
        with pytest.raises(CudaError, match="slots hold"):
            res.admit(("big",), host, 2 * MB)
        assert res.stats.refused == 1
    rt.host_free(host)


def test_a_budget_with_no_room_refuses_to_build_a_pool():
    rt = cuda_or_skip()
    tight = VramBudget.from_model(GLM, total_bytes=8 * GB, context_length=131072,
                                  layers=47, reserve_bytes=GB)
    with pytest.raises(CudaError, match="does not fit"):
        VramResidency.from_budget(rt, tight, expert_bytes=10 * MB, floor_bytes=4 * GB)


def test_transfers_are_timed_and_the_rate_is_plausible():
    rt = cuda_or_skip()
    host = rt.host_alloc(8 * MB)
    ctypes.memset(ctypes.c_void_p(host), 0x5A, 8 * MB)
    with VramResidency(rt, 8 * MB, 8) as res:
        for i in range(8):
            res.admit(("a", i), host, 8 * MB)
        s = res.stats
        assert s.transfers == 8
        assert s.bytes_transferred == 64 * MB
        assert s.transfer_seconds > 0
        assert 1 * GB < s.bytes_per_second < 200 * GB, \
            f"{s.bytes_per_second / GB:.1f} GB/s is not a believable PCIe rate"
    rt.host_free(host)


def test_releasing_frees_a_slot_without_freeing_device_memory():
    rt = cuda_or_skip()
    host = rt.host_alloc(MB)
    with VramResidency(rt, MB, 2) as res:
        res.admit(("a", 1), host, MB)
        res.release(("a", 1))
        assert ("a", 1) not in res
        res.admit(("a", 2), host, MB)
        assert res.stats.transfers == 2
    rt.host_free(host)


def test_releasing_something_absent_is_not_an_error():
    rt = cuda_or_skip()
    with VramResidency(rt, MB, 1) as res:
        res.release(("never", "there"))


def test_a_pool_needs_real_dimensions():
    rt = cuda_or_skip()
    for args in ((0, 1), (MB, 0)):
        with pytest.raises(ValueError):
            VramResidency(rt, *args)

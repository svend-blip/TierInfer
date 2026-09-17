"""Tests for byte-range reads, against a file the test writes itself."""

from __future__ import annotations

import os

import pytest

from tierinfer.index import ByteRange
from tierinfer.storage import StorageBackend, coalesce


@pytest.fixture()
def blob(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(bytes((i * 7 + 3) % 251 for i in range(64 * 1024)))
    return p


def _range(name, off, n):
    return ByteRange(name=name, file_offset=off, nbytes=n)


def test_a_read_returns_exactly_the_requested_bytes(blob):
    raw = blob.read_bytes()
    with StorageBackend(blob) as s:
        blobs, stat = s.read([_range("a", 1000, 512), _range("b", 40000, 256)])
    assert blobs[0] == raw[1000:1512]
    assert blobs[1] == raw[40000:40256]
    assert stat.nbytes == 768
    assert stat.operations == 2


def test_the_same_bytes_read_page_by_page_cost_many_more_operations(blob):
    """The comparison the project exists to win, in miniature."""
    r = [_range("a", 0, 40960)]
    with StorageBackend(blob) as s:
        _, whole = s.read(r)
        total, paged = s.read_paged(r, page=4096)
    assert total == 40960
    assert whole.operations == 1
    assert paged.operations == 10
    assert paged.mean_operation_bytes < whole.mean_operation_bytes


def test_a_short_read_is_an_error_not_a_short_answer(blob):
    past_the_end = _range("over", blob.stat().st_size - 10, 1000)
    with StorageBackend(blob) as s:
        with pytest.raises(OSError):
            s.read([past_the_end])


def test_stats_accumulate_across_reads(blob):
    with StorageBackend(blob) as s:
        s.read([_range("a", 0, 1024)])
        s.read([_range("b", 2048, 1024), _range("c", 4096, 1024)])
        assert s.stats.reads == 2
        assert s.stats.operations == 3
        assert s.stats.bytes_read == 3072
        assert s.stats.mean_operation_bytes == 1024


def test_evict_and_hint_are_safe_to_call(blob):
    """They are advisory; what matters is that they never break a read."""
    r = [_range("a", 0, 4096)]
    with StorageBackend(blob) as s:
        s.evict(r)
        s.hint_willneed(r)
        blobs, _ = s.read(r)
    assert len(blobs[0]) == 4096


def test_adjacent_ranges_coalesce_into_one_operation():
    merged = coalesce([_range("a", 0, 100), _range("b", 100, 100)])
    assert len(merged) == 1
    assert merged[0].file_offset == 0 and merged[0].nbytes == 200


def test_ranges_far_apart_are_left_alone():
    merged = coalesce([_range("a", 0, 100), _range("b", 10 << 20, 100)])
    assert len(merged) == 2


def test_coalescing_covers_every_original_byte():
    ranges = [_range("a", 0, 100), _range("b", 150, 100), _range("c", 5 << 20, 50)]
    merged = coalesce(ranges, gap=1024)
    for r in ranges:
        assert any(m.file_offset <= r.file_offset and m.end >= r.end for m in merged)


def test_out_of_order_ranges_are_sorted_before_merging():
    merged = coalesce([_range("b", 100, 100), _range("a", 0, 100)])
    assert len(merged) == 1 and merged[0].file_offset == 0

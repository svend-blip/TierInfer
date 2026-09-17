"""Tests for asynchronous expert streaming.

All of them run against a small temporary file, so the suite stays runnable
without the 56 GB model. What they bind is the behaviour that would be
expensive to discover in an inference loop: a finite pool that says when it
is empty, a cancelled load that is never mistaken for a complete one, and
buffers that come back.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.index import ByteRange  # noqa: E402
from tierinfer.storage import StorageBackend  # noqa: E402
from tierinfer.stream import (  # noqa: E402
    BufferPool, ExpertStreamer, PoolExhausted, State, StreamError,
)

SLOT = 64 * 1024


@pytest.fixture
def modelfile(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(bytes((i * 7 + 13) % 251 for i in range(1 << 20)))
    return p


@pytest.fixture
def backend(modelfile):
    with StorageBackend(modelfile) as b:
        yield b


@pytest.fixture
def streamer(backend):
    with ExpertStreamer(backend, BufferPool(SLOT, 4), workers=2) as s:
        yield s


def rng(name, offset, nbytes):
    return ByteRange(name=name, file_offset=offset, nbytes=nbytes)


# -- the pool -----------------------------------------------------------


def test_a_pool_hands_out_and_takes_back():
    pool = BufferPool(1024, 2)
    a, b = pool.acquire(), pool.acquire()
    assert pool.in_use == 2
    pool.release(a)
    assert pool.in_use == 1
    pool.release(b)
    assert pool.in_use == 0


def test_an_empty_pool_says_so_rather_than_growing():
    pool = BufferPool(1024, 1)
    pool.acquire()
    with pytest.raises(PoolExhausted):
        pool.acquire()


def test_a_pool_remembers_its_high_water_mark():
    pool = BufferPool(1024, 3)
    a, b = pool.acquire(), pool.acquire()
    pool.release(a); pool.release(b)
    assert pool.peak_in_use == 2


def test_a_pool_needs_real_dimensions():
    with pytest.raises(ValueError):
        BufferPool(0, 1)
    with pytest.raises(ValueError):
        BufferPool(1024, 0)


def test_a_foreign_buffer_is_refused():
    pool = BufferPool(1024, 1)
    with pytest.raises(StreamError):
        pool.release(bytearray(512))


# -- reading ------------------------------------------------------------


def test_a_streamed_expert_holds_the_bytes_the_file_holds(streamer, modelfile):
    want = modelfile.read_bytes()[4096:4096 + 8192]
    load = streamer.submit(("layer", 3), [rng("e", 4096, 8192)])
    got = streamer.wait(load, timeout=10)
    assert bytes(got) == want


def test_several_ranges_land_end_to_end_in_one_buffer(streamer, modelfile):
    data = modelfile.read_bytes()
    want = data[0:1024] + data[8192:8192 + 2048]
    load = streamer.submit("split", [rng("a", 0, 1024), rng("b", 8192, 2048)])
    assert bytes(streamer.wait(load, timeout=10)) == want
    assert load.nbytes == 3072


def test_many_loads_in_flight_all_arrive(streamer, modelfile):
    data = modelfile.read_bytes()
    loads = []
    for i in range(4):
        loads.append((i, streamer.submit(i, [rng(f"e{i}", i * SLOT, 4096)])))
    for i, load in loads:
        assert bytes(streamer.wait(load, timeout=10)) == data[i * SLOT:i * SLOT + 4096]
        streamer.release(load)


def test_a_load_larger_than_a_slot_is_refused_at_submission(streamer):
    with pytest.raises(StreamError, match="size the pool"):
        streamer.submit("huge", [rng("e", 0, SLOT + 1)])


def test_submitting_with_no_ranges_is_an_error(streamer):
    with pytest.raises(ValueError):
        streamer.submit("empty", [])


def test_timing_is_recorded_per_load(streamer):
    load = streamer.submit("t", [rng("e", 0, 4096)])
    streamer.wait(load, timeout=10)
    assert load.total_seconds > 0
    assert load.read_seconds > 0
    assert load.queue_seconds >= 0
    assert load.total_seconds >= load.read_seconds


def test_stats_add_up_over_several_loads(streamer):
    for i in range(3):
        load = streamer.submit(i, [rng("e", i * 4096, 4096)])
        streamer.wait(load, timeout=10)
        streamer.release(load)
    assert streamer.stats.submitted == 3
    assert streamer.stats.completed == 3
    assert streamer.stats.bytes_read == 3 * 4096
    assert streamer.stats.bytes_per_second > 0
    assert streamer.stats.mean_read_seconds > 0


# -- the states a load can end in ---------------------------------------


def test_a_view_before_the_load_is_ready_is_refused(backend):
    with ExpertStreamer(backend, BufferPool(SLOT, 1), workers=1) as s:
        load = s.submit("x", [rng("e", 0, 4096)])
        s.wait(load, timeout=10)
        s.release(load)
        assert load.state is State.RELEASED, "a released load must not still claim READY"
        with pytest.raises(StreamError):
            load.view()          # the buffer went back to the pool
        with pytest.raises(StreamError):
            s.wait(load, timeout=1)


def test_a_failed_read_is_reported_to_the_waiter_not_swallowed(backend, modelfile):
    with ExpertStreamer(backend, BufferPool(SLOT, 2), workers=1) as s:
        past_the_end = modelfile.stat().st_size + SLOT
        load = s.submit("gone", [rng("e", past_the_end, 4096)])
        with pytest.raises(StreamError):
            s.wait(load, timeout=10)
        assert load.state is State.FAILED
        assert s.stats.failed == 1


def test_a_failed_load_gives_its_buffer_back(backend, modelfile):
    pool = BufferPool(SLOT, 1)
    with ExpertStreamer(backend, pool, workers=1) as s:
        load = s.submit("gone", [rng("e", modelfile.stat().st_size + SLOT, 4096)])
        with pytest.raises(StreamError):
            s.wait(load, timeout=10)
        assert pool.in_use == 0, "a failed read leaked its slot"
        s.submit("again", [rng("e", 0, 4096)])   # the slot is usable


def test_a_cancelled_load_is_never_mistaken_for_a_complete_one(backend):
    """The pool is one slot and the worker is busy, so this load stays queued."""
    pool = BufferPool(SLOT, 2)
    with ExpertStreamer(backend, pool, workers=1) as s:
        blocker = threading.Event()
        original = s._read_one

        def slow(load):
            blocker.wait(5)
            original(load)

        s._read_one = slow
        first = s.submit("first", [rng("e", 0, 4096)])
        queued = s.submit("second", [rng("e", 4096, 4096)])
        assert s.cancel(queued) is True
        assert queued.state is State.CANCELLED
        with pytest.raises(StreamError, match="cancelled"):
            s.wait(queued, timeout=5)
        blocker.set()
        s.wait(first, timeout=10)
        assert s.stats.cancelled == 1
        assert s.stats.completed == 1


def test_cancelling_a_finished_load_changes_nothing(streamer):
    load = streamer.submit("done", [rng("e", 0, 4096)])
    streamer.wait(load, timeout=10)
    assert streamer.cancel(load) is False
    assert load.state is State.READY


def test_releasing_a_load_that_is_still_queued_is_refused(backend):
    from tierinfer.stream import Load
    with ExpertStreamer(backend, BufferPool(SLOT, 1), workers=1) as s:
        load = Load(key="k", ranges=[], nbytes=0, submitted_at=0.0)
        with pytest.raises(StreamError):
            s.release(load)


def test_the_pool_bounds_what_streaming_can_cost(backend):
    """Submitting past the pool raises instead of quietly allocating more RAM."""
    pool = BufferPool(SLOT, 2)
    with ExpertStreamer(backend, pool, workers=1) as s:
        s.submit("a", [rng("e", 0, 4096)])
        s.submit("b", [rng("e", 4096, 4096)])
        with pytest.raises(PoolExhausted):
            s.submit("c", [rng("e", 8192, 4096)])
        assert pool.nbytes == 2 * SLOT


def test_waiting_past_a_timeout_raises_rather_than_hanging(backend):
    with ExpertStreamer(backend, BufferPool(SLOT, 1), workers=1) as s:
        load = s.submit("slow", [rng("e", 0, 4096)])
        s.wait(load, timeout=10)
        load.state = State.QUEUED          # pretend it never finished
        load._done.clear()
        with pytest.raises(TimeoutError):
            s.wait(load, timeout=0.2)


def test_a_closed_streamer_refuses_new_work(backend):
    s = ExpertStreamer(backend, BufferPool(SLOT, 1), workers=1)
    s.close()
    with pytest.raises(StreamError, match="closed"):
        s.submit("x", [rng("e", 0, 4096)])
    s.close()                              # closing twice is fine


# -- the exact fallback -------------------------------------------------


def test_the_synchronous_path_reads_the_same_bytes(streamer, modelfile):
    want = modelfile.read_bytes()[2048:2048 + 4096]
    assert streamer.load_now([rng("e", 2048, 4096)]) == want


def test_the_synchronous_path_does_not_touch_the_pool(backend, modelfile):
    pool = BufferPool(SLOT, 1)
    with ExpertStreamer(backend, pool, workers=1) as s:
        s.submit("holds the only slot", [rng("e", 0, 4096)])
        assert pool.in_use == 1
        got = s.load_now([rng("e", 0, 4096)])   # must not need a slot
        assert got == modelfile.read_bytes()[:4096]

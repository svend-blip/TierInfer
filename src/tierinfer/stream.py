"""Reading experts into buffers we control, asynchronously, without mmap.

Demand paging is the thing this exists to not be. When llama.cpp maps the
model and touches an expert, the kernel decides what to read, how much of it,
and when — and the thread that touched it waits. That is fine when the model
fits in RAM and hopeless when it does not: the fault is synchronous, its size
is whatever readahead guessed, and nothing can be started before it is needed.

So reads here go through ``pread`` on a plain descriptor. Nothing is mapped,
the kernel reads exactly the bytes asked for, and the read can be issued from
a worker thread long before the token that needs it. Python releases the GIL
around ``pread``, so those threads overlap for real.

Two things are deliberately explicit.

**The buffers are pooled and finite.** ``StorageBackend.read`` allocates a
fresh ``bytes`` per range — 8.9 MB per expert on the model this was built
for, at 368 expert activations per token. A pool of fixed slots reused across
loads keeps the allocator out of the inference loop and puts a hard ceiling
on what streaming can cost in RAM. A ceiling that can be hit is a ceiling
that can be measured: exhaustion raises rather than quietly growing.

**Nothing is cancelled silently.** A prefetch that guessed wrong has to give
its slot back, and a load that was cancelled must never be mistaken for one
that completed. A handle has one terminal state and says which it is.

The synchronous path stays, and stays exact: ``load_now`` is what a
prediction miss falls back to (SCOPE goal 15). It never consults the
predictor, the cache, or anything else that could be wrong — it reads the
bytes the index names.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Iterable, Sequence

from .index import ByteRange
from .storage import StorageBackend


class StreamError(RuntimeError):
    pass


class PoolExhausted(StreamError):
    """More loads were in flight than the pool has slots."""


class State(Enum):
    QUEUED = "queued"
    READING = "reading"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RELEASED = "released"      # finished, and its buffer has gone back


# -- buffers ------------------------------------------------------------


class BufferPool:
    """A fixed number of fixed-size buffers, handed out and returned.

    The size is the largest expert the index reports, not the mean: a pool
    sized to the mean cannot hold the experts that matter most to get right.
    """

    def __init__(self, slot_bytes: int, slots: int) -> None:
        if slot_bytes <= 0 or slots <= 0:
            raise ValueError("a pool needs at least one slot of at least one byte")
        self.slot_bytes = slot_bytes
        self.slots = slots
        self._free: Queue[bytearray] = Queue()
        for _ in range(slots):
            self._free.put(bytearray(slot_bytes))
        self._lock = threading.Lock()
        self.peak_in_use = 0

    @property
    def in_use(self) -> int:
        return self.slots - self._free.qsize()

    @property
    def nbytes(self) -> int:
        return self.slot_bytes * self.slots

    def acquire(self, timeout: float | None = None) -> bytearray:
        try:
            buf = self._free.get(timeout=timeout) if timeout else self._free.get_nowait()
        except Empty:
            raise PoolExhausted(
                f"all {self.slots} buffers of {self.slot_bytes} bytes are in use; "
                "release a loaded expert before submitting another") from None
        with self._lock:
            self.peak_in_use = max(self.peak_in_use, self.in_use)
        return buf

    def release(self, buf: bytearray) -> None:
        if len(buf) != self.slot_bytes:
            raise StreamError("a buffer of the wrong size came back to the pool")
        self._free.put(buf)


# -- one load -----------------------------------------------------------


@dataclass
class Load:
    """One expert's journey from the queue to a buffer."""

    key: object
    ranges: list[ByteRange]
    nbytes: int
    submitted_at: float
    state: State = State.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    error: BaseException | None = None
    _buf: bytearray | None = None
    _done: threading.Event = field(default_factory=threading.Event)

    @property
    def queue_seconds(self) -> float:
        """How long it waited before a worker picked it up."""
        return (self.started_at - self.submitted_at) if self.started_at else 0.0

    @property
    def read_seconds(self) -> float:
        if self.started_at is None or self.finished_at is None:
            return 0.0
        return self.finished_at - self.started_at

    @property
    def total_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return self.finished_at - self.submitted_at

    def view(self) -> memoryview:
        """The bytes, without copying them. Valid until the load is released.

        A released load reports RELEASED rather than staying READY, so a use
        after release raises here instead of reaching an assert that
        ``python -O`` removes.
        """
        if self.state is not State.READY or self._buf is None:
            raise StreamError(f"load for {self.key!r} is {self.state.value}, not readable")
        return memoryview(self._buf)[:self.nbytes]


@dataclass
class StreamStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    queue_seconds: float = 0.0

    @property
    def bytes_per_second(self) -> float:
        return self.bytes_read / self.read_seconds if self.read_seconds > 0 else 0.0

    @property
    def mean_queue_seconds(self) -> float:
        return self.queue_seconds / self.completed if self.completed else 0.0

    @property
    def mean_read_seconds(self) -> float:
        return self.read_seconds / self.completed if self.completed else 0.0


# -- the streamer -------------------------------------------------------


class ExpertStreamer:
    """Issues expert reads on worker threads, into pooled buffers.

    Ordering is not promised. Several experts are in flight at once precisely
    so the device can reorder them; a caller that needs one in particular
    waits for that one.
    """

    def __init__(self, backend: StorageBackend, pool: BufferPool, *, workers: int = 4) -> None:
        if workers < 1:
            raise ValueError("a streamer needs at least one worker")
        self.backend = backend
        self.pool = pool
        self.stats = StreamStats()
        self._queue: Queue[Load | None] = Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._threads = [threading.Thread(target=self._work, daemon=True,
                                          name=f"tierinfer-stream-{i}")
                         for i in range(workers)]
        for t in self._threads:
            t.start()

    # -- submission -----------------------------------------------------

    def submit(self, key: object, ranges: Sequence[ByteRange], *,
               timeout: float | None = None) -> Load:
        """Queue an expert for reading. Raises if the pool has no slot."""
        if self._closed:
            raise StreamError("streamer is closed")
        rs = list(ranges)
        if not rs:
            raise ValueError(f"no byte ranges for {key!r}")
        nbytes = sum(r.nbytes for r in rs)
        if nbytes > self.pool.slot_bytes:
            raise StreamError(
                f"{key!r} needs {nbytes} bytes but the pool's slots hold "
                f"{self.pool.slot_bytes}; size the pool to the largest expert")
        buf = self.pool.acquire(timeout=timeout)
        load = Load(key=key, ranges=rs, nbytes=nbytes, submitted_at=time.perf_counter())
        load._buf = buf
        with self._lock:
            self.stats.submitted += 1
        self._queue.put(load)
        return load

    def wait(self, load: Load, timeout: float | None = None) -> memoryview:
        """Block until this load is readable, or raise what stopped it."""
        if not load._done.wait(timeout=timeout):
            raise TimeoutError(f"load for {load.key!r} did not finish in {timeout}s")
        if load.state is State.FAILED:
            raise StreamError(f"load for {load.key!r} failed") from load.error
        if load.state is State.CANCELLED:
            raise StreamError(f"load for {load.key!r} was cancelled")
        if load.state is State.RELEASED:
            raise StreamError(f"load for {load.key!r} was already released")
        return load.view()

    def cancel(self, load: Load) -> bool:
        """Give up on a load that has not started. True when it was still queued."""
        with self._lock:
            if load.state is not State.QUEUED:
                return False
            load.state = State.CANCELLED
            self.stats.cancelled += 1
        self._return_buffer(load)
        load.finished_at = time.perf_counter()
        load._done.set()
        return True

    def release(self, load: Load) -> None:
        """Return a finished load's buffer to the pool. The view dies with it."""
        if load.state is State.QUEUED or load.state is State.READING:
            raise StreamError(f"load for {load.key!r} is still {load.state.value}")
        self._return_buffer(load)
        if load.state is State.READY:
            load.state = State.RELEASED

    def _return_buffer(self, load: Load) -> None:
        buf, load._buf = load._buf, None
        if buf is not None:
            self.pool.release(buf)

    # -- the workers ----------------------------------------------------

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._read_one(item)

    def _read_one(self, load: Load) -> None:
        with self._lock:
            if load.state is not State.QUEUED:
                return                     # cancelled between queue and pickup
            load.state = State.READING
            load.started_at = time.perf_counter()
        try:
            view = memoryview(load._buf)
            written = 0
            for r in load.ranges:
                got = os.preadv(self.backend.fd_for(r), [view[written:written + r.nbytes]],
                                r.file_offset)
                if got != r.nbytes:
                    raise OSError(f"short read on {r.name}: {got} of {r.nbytes} bytes")
                written += got
            load.finished_at = time.perf_counter()
            with self._lock:
                load.state = State.READY
                self.stats.completed += 1
                self.stats.bytes_read += written
                self.stats.read_seconds += load.read_seconds
                self.stats.queue_seconds += load.queue_seconds
        except BaseException as exc:       # noqa: BLE001 — recorded, then re-raised to the waiter
            load.error = exc
            load.finished_at = time.perf_counter()
            with self._lock:
                load.state = State.FAILED
                self.stats.failed += 1
            self._return_buffer(load)
        finally:
            load._done.set()

    # -- the exact path -------------------------------------------------

    def load_now(self, ranges: Sequence[ByteRange]) -> bytes:
        """Read these bytes synchronously, consulting nothing.

        This is the fallback a prediction miss takes (SCOPE goal 15). It does
        not touch the pool, the queue or the workers, so it cannot be blocked
        by them and cannot be wrong about what it returns.
        """
        blobs, _ = self.backend.read(list(ranges))
        return b"".join(blobs)

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Stop the workers. Anything still queued fails rather than hangs.

        The workers drain what is ahead of the sentinels, so in the ordinary
        case the queue is empty by the time they exit. If a join times out
        and something is still queued, its waiter would otherwise block
        forever on an event nobody will set.
        """
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=5.0)
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            if item is None:
                continue
            with self._lock:
                if item.state is not State.QUEUED:
                    continue
                item.state = State.FAILED
                item.error = StreamError("streamer closed before this load started")
                self.stats.failed += 1
            item.finished_at = time.perf_counter()
            self._return_buffer(item)
            item._done.set()

    def __enter__(self) -> "ExpertStreamer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

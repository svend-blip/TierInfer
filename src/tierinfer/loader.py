"""The loader: answer a running llama.cpp's page faults with whole experts.

`docs/superpowers/specs/2026-09-18-llama-loader-design.md` says why it is
built this way; this is the half that lives in TierInfer's own process. The
other half, `tools/uffd/tierinfer_mmap.c`, is preloaded into an unmodified
llama.cpp and does three things: it hands this server an anonymous,
userfaultfd-registered region in place of each model file's mapping, it
sends every ``ffn_moe_topk`` routing row here, and it drops the pages this
server tells it to. Everything that decides is here.

**What a fault is.** llama.cpp's compute thread touched a page nobody had
materialised. The page belongs to a byte range of a model file, and the
index says what that range is: a slab of one routed expert, or a piece of
the floor (attention, norms, embeddings). For an expert the answer is the
*whole expert* — all three projections, 26–31 MB on the 480B — copied into
the process with ``UFFDIO_COPY`` from a buffer this server just read with
``pread``. One fault per expert instead of seven thousand; the coalescing
the baseline experiment asked for, done by construction.

**What the RAM tier is.** The set of experts currently materialised in the
llama.cpp process. This server holds no second copy. The set is bounded by
a byte budget; over it, `ExpertCache`'s policy (recency-led, LRU-equal on
measured routing) names a victim and the shim ``madvise(MADV_DONTNEED)``s
its slabs, so the next touch faults again and is decided anew.

**What a hit is.** A routed expert that is already materialised when its
routing arrives. Not a page-cache guess: the server knows exactly which
experts it has copied and not yet evicted, and the routing names exactly
which the token needs.

**What prefetch is.** ``UFFDIO_COPY`` before the touch, for experts the
predictor names for the next layer. With real compute between a layer's
routing and its FFN, and between layers, prefetch has lead time here that
the replay could only emulate. Whether it earns anything is measured, not
assumed; depth 0 turns it off.

**Correctness.** The bytes copied for a range are the file's bytes for that
range, read through `StorageBackend` on the index's own offsets, including
the few edge bytes of a neighbouring tensor that share a page with a slab.
A read that fails is retried once through the exact path; if that fails the
server logs and stops rather than answer with anything else, and the
faulting thread waits — a model whose bytes cannot be read does not run,
and never runs wrong. The acceptance test is identical greedy tokens
against native.
"""

from __future__ import annotations

import array
import bisect
import ctypes
import ctypes.util
import errno
import faulthandler
import fcntl
import mmap
import os
import select
import signal
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .cache import ExpertCache
from .index import ByteRange, ModelIndex
from .predict import AdaptiveBlend, Frequency, Persistence, Predictor, Transition
from .storage import StorageBackend
from .telemetry import Telemetry
from .tracker import ExpertKey, ExpertTracker

PAGE = mmap.PAGESIZE
GB = 1024 ** 3
MB = 1024 ** 2

# -- userfaultfd ioctls (x86_64 layout; struct sizes from <linux/userfaultfd.h>)
UFFD_API = 0xAA
_IOC = lambda d, t, n, s: (d << 30) | (s << 16) | (t << 8) | n  # noqa: E731
UFFDIO_COPY = _IOC(3, UFFD_API, 0x03, 40)        # struct uffdio_copy {dst, src, len, mode, copy}
UFFDIO_WAKE = _IOC(2, UFFD_API, 0x02, 16)        # struct uffdio_range {start, len}
UFFD_EVENT_PAGEFAULT = 0x12
_MSG = 32                                          # sizeof(struct uffd_msg)

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class LoaderError(RuntimeError):
    pass


# -- what a layout source has to answer -------------------------------------


class GGUFSource:
    """`tierinfer.index.ModelIndex` as the loader sees it."""

    kind = "gguf"

    def __init__(self, index) -> None:
        self.index = index

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self.index.gguf.files)

    @property
    def moe_layers(self) -> list[int]:
        return self.index.moe_layers

    @property
    def expert_count(self) -> int:
        return self.index.expert_count

    def expert_ranges(self, layer: int, expert: int) -> list[ByteRange]:
        return list(self.index.expert(layer, expert).ranges)

    def floor_ranges(self) -> list[ByteRange]:
        expert_tensors = set()
        for layer in self.index.moe_layers:
            for r in self.index.expert(layer, 0).ranges:
                expert_tensors.add(r.name.split("#")[0])
        return [ByteRange(t.name, t.file_offset, t.nbytes, t.path)
                for t in self.index.gguf.tensors if t.name not in expert_tensors]

    def expert_nbytes_max(self) -> int:
        return self.index.expert_nbytes_max()

    def describe(self) -> str:
        return str(self.index.gguf.path)

    def physical(self, logical_off: int, nbytes: int) -> list[ByteRange]:
        raise LoaderError("a GGUF mapping is a file, not a logical region")


class FTWSource:
    """`tierinfer.ftw.FTWIndex` as the loader sees it: FreeToken's checkpoint."""

    kind = "ftw"

    def __init__(self, ftw) -> None:
        self.ftw = ftw

    @property
    def files(self) -> tuple[Path, ...]:
        return self.ftw.files

    @property
    def moe_layers(self) -> list[int]:
        return self.ftw.moe_layers

    @property
    def expert_count(self) -> int:
        return self.ftw.expert_count

    def expert_ranges(self, layer: int, expert: int) -> list[ByteRange]:
        return self.ftw.expert_rows(layer, expert)

    def floor_ranges(self) -> list[ByteRange]:
        out: list[ByteRange] = []
        for e in self.ftw.floor_entries():
            out += self.ftw.physical(e.name, e.global_off, e.nbytes)
        return out

    def expert_nbytes_max(self) -> int:
        return max(self.ftw.expert_nbytes(l) for l in self.ftw.moe_layers) if self.ftw.moe_layers else 0

    def describe(self) -> str:
        return str(self.ftw.directory)

    def physical(self, logical_off: int, nbytes: int) -> list[ByteRange]:
        return self.ftw.physical(f"logical:{logical_off}", logical_off, nbytes)

    def logical_key(self, logical_off: int):
        hit = self.ftw.logical_to_key(logical_off)
        return hit[0] if hit else None


def as_source(index_or_source):
    if hasattr(index_or_source, "expert_ranges"):
        return index_or_source
    if hasattr(index_or_source, "gguf"):
        return GGUFSource(index_or_source)
    if hasattr(index_or_source, "expert_rows"):
        return FTWSource(index_or_source)
    raise LoaderError(f"cannot serve a {type(index_or_source).__name__}: not a GGUF index or an FTW index")


# -- the layout of a mapping ------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A byte range of one file and what it is: an expert's slab or floor."""

    start: int
    end: int
    key: object          # (layer, expert) for a routed expert; ("floor", tensor) otherwise
    name: str


class FileLayout:
    """Every region of one model file, addressable by offset in O(log n).

    Built from a layout source (GGUF or FTW). For a *logical* mapping — a
    buffer standing in for a slice of an FTW region rather than a file —
    ``logical_off``/``logical_len`` select the slice, and region offsets
    are buffer offsets.
    """

    def __init__(self, source, path: Path | None, *, logical_off: int | None = None,
                 logical_len: int = 0) -> None:
        source = as_source(source)
        self.source = source
        self.path = path
        self.logical_off = logical_off
        self.size = path.stat().st_size if path is not None else logical_len
        regions: list[Region] = []
        if logical_off is None:
            for layer in source.moe_layers:
                for e in range(source.expert_count):
                    for r in source.expert_ranges(layer, e):
                        if r.path == path:
                            regions.append(Region(r.file_offset, r.end, (layer, e), r.name))
            for r in source.floor_ranges():
                if r.path == path:
                    regions.append(Region(r.file_offset, r.end, ("floor", r.name), r.name))
        else:
            # buffer offset b is logical offset logical_off + b; walk the
            # region's entries and keep what falls inside the slice
            ftw = source.ftw
            lo, hi = logical_off, logical_off + logical_len
            for e in ftw.entries:
                if e.end <= lo or e.global_off >= hi:
                    continue
                layer_bank = ftw.logical_to_key(e.global_off)
                if layer_bank and layer_bank[0][0] != "floor":
                    layer = layer_bank[0][0]
                    row = e.nbytes // ftw.num_experts
                    for x in range(ftw.num_experts):
                        a, b = e.global_off + x * row, e.global_off + (x + 1) * row
                        if b <= lo or a >= hi:
                            continue
                        regions.append(Region(max(a, lo) - lo, min(b, hi) - lo, (layer, x), f"{e.name}#expert{x}"))
                else:
                    regions.append(Region(max(e.global_off, lo) - lo, min(e.end, hi) - lo, ("floor", e.name), e.name))
        regions.sort(key=lambda r: r.start)
        self.regions = regions
        self._starts = [r.start for r in regions]
        #: All three slabs of an expert, by key, for whole-expert service.
        self.by_key: dict[object, list[Region]] = defaultdict(list)
        for r in regions:
            self.by_key[r.key].append(r)

    def at(self, offset: int) -> Region | None:
        i = bisect.bisect_right(self._starts, offset) - 1
        if i < 0:
            return None
        r = self.regions[i]
        return r if r.start <= offset < r.end else None

    def owner_of_page(self, page_start: int) -> Region | None:
        """Which region a fault on this page is for.

        The kernel reports faults page-aligned, and slab boundaries are
        32-byte aligned, so the first page of a slab almost always also
        holds the tail of the slab before it. A slab that is resident has
        all its pages, and an evicted one keeps its edge pages (eviction
        drops interior pages only), so a fault on a straddled page cannot be
        for the slab that *ends* in it — it is for the one that *starts* in
        it. When nothing starts in the page, the region spanning it owns it.
        """
        head = self.at(page_start)
        tail = self.at(page_start + PAGE - 1)
        if tail is not None and tail is not head and tail.start > page_start:
            return tail
        if head is not None:
            return head
        # the page starts in padding: whatever begins inside it, if anything
        i = bisect.bisect_left(self._starts, page_start)
        if i < len(self.regions) and self.regions[i].start < page_start + PAGE:
            return self.regions[i]
        return None


@dataclass
class Mapping:
    """One anonymous region in the client, standing in for one model file."""

    path: Path
    base: int
    length: int
    uffd: int
    layout: FileLayout
    pid: int
    pagemap_fd: int = -1          # /proc/<pid>/pagemap, when readable
    dead: bool = False
    logical_off: int | None = None    # set when the mapping is a slice of an FTW region, not a file
    holes: list[tuple[int, int]] = field(default_factory=list)   # [a, b) offsets the client has unmapped
    gone: set = field(default_factory=set)                        # keys whose every byte lies in a hole

    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.base + self.length

    def in_hole(self, a: int, b: int) -> bool:
        """Does [a, b) touch memory the client has unmapped?"""
        return any(a < hb and b > ha for ha, hb in self.holes)

    def live_pieces(self, a: int, b: int) -> list[tuple[int, int]]:
        """[a, b) minus the holes, in order."""
        pieces = [(a, b)]
        for ha, hb in self.holes:
            nxt = []
            for x, y in pieces:
                if y <= ha or x >= hb:
                    nxt.append((x, y))
                else:
                    if x < ha:
                        nxt.append((x, ha))
                    if y > hb:
                        nxt.append((hb, y))
            pieces = nxt
        return pieces

    def open_pagemap(self) -> None:
        try:
            self.pagemap_fd = os.open(f"/proc/{self.pid}/pagemap", os.O_RDONLY)
        except OSError:
            self.pagemap_fd = -1

    def page_present(self, offset: int) -> bool | None:
        """Whether the client's page at this file offset is present, from
        /proc/<pid>/pagemap; ``None`` when that cannot be read here."""
        if self.pagemap_fd < 0:
            return None
        try:
            raw = os.pread(self.pagemap_fd, 8, ((self.base + offset) // PAGE) * 8)
        except OSError:
            return None
        if len(raw) != 8:
            return None
        return bool(int.from_bytes(raw, "little") >> 63 & 1)


# -- prefetch scheduling ------------------------------------------------------


class Priority:
    """The four classes the scope names (7.7). Lower serves first.

    REQUIRED_NOW is a demand fault and never queues — the fault workers serve
    it directly — but it is a class so telemetry can say what share of reads
    were demand and what share speculation."""

    REQUIRED_NOW = 0
    HIGH_CONFIDENCE_NEXT = 1     # the next layer's top guesses
    PROBABLE_NEXT = 2            # further layers, or lower-ranked guesses
    BACKGROUND_HOT = 3           # hot experts by frequency, when the tier has room and nothing else waits

    NAMES = {0: "required_now", 1: "high_confidence_next", 2: "probable_next", 3: "background_hot"}


@dataclass(order=True)
class _Job:
    priority: int
    seq: int
    key: object = field(compare=False)
    layer: int = field(compare=False)
    token: int = field(compare=False)


class PrefetchScheduler:
    """A bounded priority queue of speculative reads with supersession.

    ``serve(keys)`` is called from the worker threads with a batch of keys
    of one layer, so the server can read adjacent slabs together
    (coalescing). Jobs for a layer whose routing has already arrived are
    dropped unserved and counted as superseded: reading them now would be
    a demand read that is already happening elsewhere, or waste.
    """

    def __init__(self, serve: Callable[[list], None], *, workers: int = 4,
                 capacity: int = 64) -> None:
        import queue
        self.serve = serve
        self.capacity = capacity
        self._q: "queue.PriorityQueue[_Job]" = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._done_layers: dict[int, int] = {}      # token -> highest layer routed
        self.enqueued = 0
        self.served = 0
        self.dropped_full = 0
        self.superseded = 0
        self.by_class = {p: 0 for p in Priority.NAMES}
        self._threads = [threading.Thread(target=self._loop, daemon=True,
                                          name=f"tierinfer-prefetch-{i}") for i in range(workers)]
        for t in self._threads:
            t.start()

    def submit(self, key, *, layer: int, token: int, priority: int) -> bool:
        with self._lock:
            if self._q.qsize() >= self.capacity:
                self.dropped_full += 1
                return False
            self._seq += 1
            self._q.put(_Job(priority, self._seq, key, layer, token))
            self.enqueued += 1
            self.by_class[priority] = self.by_class.get(priority, 0) + 1
        return True

    def routed(self, layer: int, token: int) -> None:
        """Routing for this layer has arrived: older jobs for it are stale."""
        with self._lock:
            self._done_layers[token] = max(self._done_layers.get(token, -1), layer)
            for t in [t for t in self._done_layers if t < token - 1]:
                del self._done_layers[t]

    def _stale(self, job: _Job) -> bool:
        done = self._done_layers.get(job.token, -1)
        return job.layer <= done or job.token < max(self._done_layers, default=job.token) - 1

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job.key is None:                    # the stop sentinel
                return
            batch = [job]
            # gather what else is queued for the same layer, to serve together
            try:
                while len(batch) < 8:
                    nxt = self._q.get_nowait()
                    if nxt.key is None:
                        self._q.put(nxt)
                        break
                    if nxt.layer == job.layer and nxt.token == job.token:
                        batch.append(nxt)
                    else:
                        self._q.put(nxt)
                        break
            except Exception:  # noqa: BLE001 — queue.Empty
                pass
            with self._lock:
                live = [j for j in batch if not self._stale(j)]
                self.superseded += len(batch) - len(live)
            if live:
                try:
                    self.serve([j.key for j in live])
                    self.served += len(live)
                except Exception as e:  # noqa: BLE001 — speculation may fail; demand is elsewhere
                    print(f"tierinfer-loader: prefetch batch failed: {e}", file=sys.stderr, flush=True)

    def close(self) -> None:
        for _ in self._threads:
            self._q.put(_Job(10 ** 6, 0, None, -1, -1))


# -- the residency tier -----------------------------------------------------


class _ResidencyCache(ExpertCache):
    """ExpertCache that tells someone when it evicts. Payloads are ``None``:
    the bytes live in the client's mapping, the policy lives here."""

    def __init__(self, capacity: int, tracker: ExpertTracker, on_evict: Callable[[object], None]) -> None:
        super().__init__(capacity, tracker)
        self._on_evict = on_evict

    def _drop(self, key, *, evicted: bool) -> None:
        super()._drop(key, evicted=evicted)
        if evicted:
            self._on_evict(key)


@dataclass
class LoaderStats:
    faults: int = 0
    faults_expert: int = 0
    faults_floor: int = 0
    faults_duplicate: int = 0           # another thread was already serving it
    faults_resident: int = 0            # arrived for an expert already materialised; woken, not re-copied
    faults_repaired: int = 0            # resident yet faulting repeatedly: copied again
    bytes_copied: int = 0
    copy_seconds: float = 0.0
    read_seconds: float = 0.0
    routed: int = 0
    hits: int = 0                       # routed and already materialised
    misses: int = 0                     # routed and not materialised (a fault follows)
    prefetch_issued: int = 0
    prefetch_useful: int = 0            # routed while materialised by prefetch, before any fault
    prefetch_late: int = 0              # routed while a prefetch for it was still in flight
    prefetch_wasted: int = 0            # evicted without ever having been routed to
    evictions: int = 0
    evict_bytes: int = 0
    tokens: int = 0
    read_retries: int = 0
    copy_eagain: int = 0
    wakes: int = 0
    unmapped_pages: int = 0
    coalesced_reads: int = 0            # one read that covered a run of consecutive experts
    coalesced_experts: int = 0          # experts served through such reads
    repeat_faults: int = 0
    unmaps: int = 0                     # UNMAP messages from clients
    unmapped_bytes: int = 0
    forgotten: int = 0                  # resident experts whose bytes the client unmapped

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class LoaderServer:
    """Accepts mappings from preloaded llama.cpp processes and serves them."""

    def __init__(self, index, *, ram_bytes: int, workers: int = 8, depth: int = 0,
                 telemetry: Telemetry | None = None, floor_chunk: int = 16 * MB,
                 predictor: Predictor | None = None, verbose: bool = True,
                 drop_page_cache: bool = True) -> None:
        self.source = as_source(index)
        self.index = index
        self.drop_page_cache = drop_page_cache
        self.ram_bytes = ram_bytes
        self.workers = workers
        self.depth = depth
        self.floor_chunk = floor_chunk
        self.verbose = verbose
        self.tel = telemetry
        self.backend = StorageBackend(self.source.files)
        self.layouts: dict[object, FileLayout] = {}
        self.mappings: list[Mapping] = []
        self.tracker = ExpertTracker(window=128)
        self.predictor = predictor or AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16)
        self.cache = _ResidencyCache(ram_bytes, self.tracker, self._evicted)
        self.stats = LoaderStats()
        self._evict_conns: dict[int, socket.socket] = {}       # pid -> evict socket
        self._lock = threading.RLock()
        self._serving: dict[object, threading.Event] = {}      # key -> done event
        self._prefetched: set = set()
        self._token_faulted: set = set()      # experts faulted in since the last token boundary
        self._after_faulted: set = set()      # the previous token's, for a ROUTED burst that crosses the boundary                          # keys materialised by prefetch, not yet routed
        self._floor_done: set = set()                          # (path, chunk index)
        self._repeats: dict[int, int] = {}                     # page offset -> consecutive faults seen
        self._resident_refaults: dict[object, int] = {}        # key -> faults while already resident
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tierinfer-fault")
        self.scheduler = PrefetchScheduler(self._serve_batch, workers=max(1, workers // 2),
                                           capacity=max(16, 8 * max(depth, 1)))
        # routing state for the current token
        self._last_layer = -1
        self._sofar: dict[int, list[int]] = {}
        self._token_keys: set = set()
        self._token_started = time.perf_counter()
        self._token_stats = None
        self.debug = bool(os.environ.get("TIERINFER_DEBUG"))
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
            signal.signal(signal.SIGUSR2, lambda *_: self._say(f"stats {self.stats} resident {len(self.cache)} "
                                                                  f"({self.cache.used_bytes / GB:.1f} GB) "
                                                                  f"serving {len(self._serving)}"))
        except (AttributeError, ValueError, RuntimeError):
            pass                       # not the main thread, or no such signal here

    # -- lifecycle ----------------------------------------------------------

    def serve(self, sock_path: str | Path) -> None:
        """Listen forever (until ``stop``) for preloaded clients."""
        sock_path = Path(sock_path)
        if sock_path.exists():
            sock_path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(8)
        srv.settimeout(0.5)
        self._say(f"listening on {sock_path}; RAM tier {self.ram_bytes / GB:.1f} GB "
                  f"({self.ram_bytes // max(1, self.source.expert_nbytes_max())} experts), "
                  f"{self.workers} fault workers, prefetch depth {self.depth}")
        if self.tel:
            self.tel.open_run(model=self.source.describe(), layout=self.source.kind, ram_bytes=self.ram_bytes,
                              workers=self.workers, depth=self.depth, files=len(self.source.files))
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._client, args=(conn,), daemon=True,
                                 name="tierinfer-client").start()
        finally:
            srv.close()
            try:
                sock_path.unlink()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        self._pool.shutdown(wait=False)
        self.scheduler.close()
        if self.tel:
            self.tel.close_run(**self._snapshot_values())
            self.tel.close()
        self.backend.close()

    # -- the two connections a client makes ----------------------------------

    def _client(self, conn: socket.socket) -> None:
        f = conn.makefile("rb", buffering=0)
        hello = _readline(conn)
        parts = hello.split()
        if len(parts) != 3 or parts[0] != "HELLO":
            self._say(f"refusing a client that said {hello!r}")
            conn.close()
            return
        kind, pid = parts[1], int(parts[2])
        if kind == "probe":
            conn.close()                        # a liveness check from a harness
            return
        if kind == "evict":
            with self._lock:
                self._evict_conns[pid] = conn
            self._say(f"pid {pid}: eviction channel open")
            return                              # kept open; written to by _evicted
        self._say(f"pid {pid}: control channel open")
        reader = _LineReader(conn)
        try:
            while not self._stop.is_set():
                line, fds = reader.line()
                if line is None:
                    break
                if line.startswith("MAP "):
                    self._on_map(conn, pid, line, fds)
                elif line.startswith("ROUTE "):
                    self._on_route(line)
                elif line.startswith("ROUTED "):
                    self._on_route(line, after=True)
                elif line.startswith("UNMAP "):
                    self._on_unmap(pid, line)
        finally:
            self._say(f"pid {pid}: control channel closed")
            f.close()

    def _on_map(self, conn: socket.socket, pid: int, line: str, fds: list[int]) -> None:
        # MAP <path-or-tag> <base> <len> [<logical_off>]
        parts = line.split()
        path_s, base_s, len_s = parts[1], parts[2], parts[3]
        logical_off = int(parts[4]) if len(parts) > 4 else None
        if not fds:
            conn.sendall(b"NO no descriptor attached\n")
            return
        uffd = fds[0]
        if logical_off is None:
            path = Path(path_s)
            if path not in {Path(p) for p in self.source.files}:
                conn.sendall(b"NO not a file of this model\n")
                os.close(uffd)
                return
            layout = self.layouts.get(path)
            if layout is None:
                layout = self.layouts[path] = FileLayout(self.source, path)
        else:
            if self.source.kind != "ftw":
                conn.sendall(b"NO a logical mapping needs an FTW layout\n")
                os.close(uffd)
                return
            path = Path(path_s)             # a tag naming the buffer, kept for messages
            layout = FileLayout(self.source, None, logical_off=logical_off, logical_len=int(len_s))
        m = Mapping(path=path, base=int(base_s, 16), length=int(len_s), uffd=uffd, layout=layout, pid=pid,
                    logical_off=logical_off)
        m.open_pagemap()
        with self._lock:
            self.mappings.append(m)
            new_uffd = all(x.uffd != uffd for x in self.mappings[:-1])
        if new_uffd:
            for i in range(self.workers):
                threading.Thread(target=self._fault_loop, args=(uffd,), daemon=True,
                                 name=f"tierinfer-uffd-{i}").start()
        conn.sendall(b"OK\n")
        self._say(f"pid {pid}: serving {path.name} ({m.length / GB:.2f} GB at {m.base:#x}), "
                  f"{len(layout.regions)} regions")
        if self.tel:
            self.tel.event("mapping", pid=pid, path=str(path), length=m.length)

    # -- faults -------------------------------------------------------------

    def _fault_loop(self, uffd: int) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([uffd], [], [], 0.5)
            except (OSError, ValueError):
                return
            if not r:
                continue
            try:
                data = os.read(uffd, _MSG * 64)
            except BlockingIOError:
                continue
            except OSError as e:
                if e.errno == errno.EBADF:
                    return
                raise
            for off in range(0, len(data) - _MSG + 1, _MSG):
                event = data[off]
                if event != UFFD_EVENT_PAGEFAULT:
                    self._say(f"uffd event {event:#x} ignored")
                    continue
                flags, addr = struct.unpack_from("<QQ", data, off + 8)
                try:
                    self._fault(addr, flags, uffd)
                except Exception as e:        # noqa: BLE001 — a fault left unanswered hangs the client
                    self._say(f"FAULT AT {addr:#x} NOT SERVED: {type(e).__name__}: {e}")
                    raise

    def _fault(self, addr: int, flags: int = 0, uffd: int = -1) -> None:
        m = self._mapping_for(addr)
        if m is None:
            self._say(f"fault at {addr:#x} outside every mapping — cannot serve")
            return
        offset = (addr - m.base) & ~(PAGE - 1)
        region = m.layout.owner_of_page(offset)
        self.stats.faults += 1
        n = self._repeats.get(offset, 0) + 1
        self._repeats[offset] = n
        if n > 64:
            self.stats.repeat_faults += 1
            raise LoaderError(f"page at file offset {offset} of {m.path.name} has faulted {n} times "
                              f"without becoming present (region {region.key if region else None}); "
                              "refusing to spin — the client is left waiting rather than lied to")
        if len(self._repeats) > 4096:
            self._repeats.clear()
        if self.debug:
            self._say(f"fault {addr:#x} flags {flags:#x} off {offset} -> {region.key if region else None}")
        if m.dead:
            return
        if region is None:
            # Padding between tensors, or the tail past EOF: the file's bytes
            # there (or zeros past its end) — never anything else.
            self._copy_pages(m, offset, offset + PAGE, lambda a, b: self._file_bytes(m, a, b))
        elif region.key[0] == "floor":
            self.stats.faults_floor += 1
            self._serve_floor(m, offset)
        else:
            self.stats.faults_expert += 1
            self._serve_expert(m, region.key, why="fault", offset=offset)
        # Wake the faulting page explicitly. A copy wakes the range it copied;
        # a page found already present is not copied and so wakes nobody, and
        # a thread that faulted on it in the window between its fault and the
        # copy that made it present would otherwise wait forever.
        self._wake(m, offset, offset + PAGE)

    def _wake(self, m: Mapping, a: int, b: int) -> None:
        req = struct.pack("<QQ", m.base + a, b - a)
        try:
            fcntl.ioctl(m.uffd, UFFDIO_WAKE, req)
            self.stats.wakes += 1
        except OSError as e:
            if e.errno not in (errno.ENOENT, errno.ESRCH):
                self._say(f"UFFDIO_WAKE {m.base + a:#x}: {e}")

    def _mapping_for(self, addr: int) -> Mapping | None:
        for m in self.mappings:
            if m.contains(addr):
                return m
        return None

    # -- serving ------------------------------------------------------------

    def _serve_expert(self, m: Mapping, key: object, *, why: str, offset: int = -1) -> None:
        """Materialise all slabs of one expert in the client, once."""
        with self._lock:
            if key in self.cache:
                if why != "fault":
                    return
                # Resident by our books, yet a fault arrived. Almost always
                # this is a queued event from a thread that faulted while the
                # copy was in flight — llama.cpp's compute threads touch one
                # expert's pages in parallel — and its page is present now;
                # the explicit wake after this call releases it. The second
                # live run guessed at that with a counter and re-copied
                # every third such fault (36 000 repairs in 30 tokens, 8.9 GB
                # a token for 1 GB of misses). Now the client's own page
                # table answers: present → wake only; absent → copy.
                present = m.page_present(offset) if offset >= 0 else None
                if present is None:
                    n = self._resident_refaults.get(key, 0) + 1
                    self._resident_refaults[key] = n
                    present = n < 3
                    if not present:
                        self._resident_refaults[key] = 0
                if present:
                    self.stats.faults_resident += 1
                    return
                self.stats.faults_repaired += 1
            ev = self._serving.get(key)
            if ev is not None:
                waiting = True
            else:
                ev = self._serving[key] = threading.Event()
                waiting = False
        if waiting:
            self.stats.faults_duplicate += 1
            ev.wait(timeout=120)
            return
        try:
            # An expert's slabs are in one file mapping for a GGUF; for an FTW
            # model each bank is its own buffer, so the same key has regions
            # in several of the client's mappings. Every one is materialised:
            # the expert is resident whole or not at all, whichever bank it
            # was first touched through.
            nbytes = 0
            for mm in self._mappings_with(key, m.pid):
                for r in mm.layout.by_key[key]:
                    a = r.start & ~(PAGE - 1)
                    b = min((r.end + PAGE - 1) & ~(PAGE - 1), (mm.length + PAGE - 1) & ~(PAGE - 1))
                    self._copy_pages(mm, a, b, lambda x, y, mm=mm: self._file_bytes(mm, x, y))
                    nbytes += r.end - r.start
            with self._lock:
                admitted = self.cache.put(key, None, nbytes)
                if why == "fault":
                    self._token_faulted.add(key)
                if why == "prefetch":
                    self._prefetched.add(key)
                if not admitted:
                    self._say(f"expert {key} does not fit the tier at all ({nbytes} bytes)")
        finally:
            with self._lock:
                self._serving.pop(key, None)
            ev.set()

    def _mappings_with(self, key: object, pid: int) -> list[Mapping]:
        """The live mappings of this client that hold slabs of this expert."""
        return [x for x in self.mappings if x.pid == pid and not x.dead and key in x.layout.by_key]

    def _serve_floor(self, m: Mapping, offset: int) -> None:
        """Materialise the aligned chunk of floor around ``offset``, pinned."""
        chunk = offset // self.floor_chunk
        tag = (m.path, chunk)
        with self._lock:
            if tag in self._floor_done:
                present = m.page_present(offset)
                if present is None:
                    n = self._resident_refaults.get(tag, 0) + 1
                    self._resident_refaults[tag] = n
                    present = n < 3
                    if not present:
                        self._resident_refaults[tag] = 0
                if present:
                    self.stats.faults_resident += 1
                    return
                self._floor_done.discard(tag)
                self.stats.faults_repaired += 1
            ev = self._serving.get(tag)
            if ev is not None:
                waiting = True
            else:
                ev = self._serving[tag] = threading.Event()
                waiting = False
        if waiting:
            ev.wait(timeout=120)
            return
        try:
            a = chunk * self.floor_chunk
            b = min(a + self.floor_chunk, (m.length + PAGE - 1) & ~(PAGE - 1))
            self._copy_pages(m, a, b, lambda x, y: self._file_bytes(m, x, y))
            with self._lock:
                self._floor_done.add(tag)
                self.cache.put(("floor", str(m.path), chunk), None, b - a, pinned=True)
        finally:
            with self._lock:
                self._serving.pop(tag, None)
            ev.set()

    def _file_bytes(self, m: Mapping, a: int, b: int) -> bytes:
        """The bytes behind [a, b) of the mapping, zero-padded past its end.

        For a file mapping that is the file; for a logical slice of an FTW
        region it is the shard bytes the FTW index names for those logical
        offsets. Exact path either way, retried once.
        """
        end = min(b, m.layout.size)
        if m.logical_off is None:
            wants = [ByteRange(name=f"{m.path.name}:{a}", file_offset=a, nbytes=end - a, path=m.path)]
        else:
            wants = self.source.physical(m.logical_off + a, end - a)
        t0 = time.perf_counter()
        try:
            blobs, _ = self.backend.read(wants)
        except OSError as e:
            self.stats.read_retries += 1
            self._say(f"read of {wants[0].name} failed ({e}); retrying once")
            blobs, _ = self.backend.read(wants)
        self.stats.read_seconds += time.perf_counter() - t0
        if self.drop_page_cache:
            # The bytes now live in the client's mapping; a second copy in the
            # page cache would be RAM the tier does not account for, and would
            # make an eviction a lie (the next fault would be served from RAM).
            try:
                self.backend.evict(wants)
            except OSError as e:
                self._say(f"could not drop page cache behind {wants[0].name}: {e}")
        data = b"".join(blobs) if len(blobs) > 1 else blobs[0]
        if end < b:
            data += bytes(b - end)
        return data

    def _copy_pages(self, m: Mapping, a: int, b: int, source: Callable[[int, int], bytes]) -> int:
        """UFFDIO_COPY the file's bytes for pages [a, b) of the mapping.

        Pages already present (a neighbour's edge, a race with another
        worker) return EEXIST; they are skipped a page at a time, because a
        present page is by construction already the right bytes.

        A range that reaches into memory the client has unmapped returns
        ENOENT *before anything is copied* — the kernel checks the whole
        destination first. llama.cpp unmaps the fragments of a file no used
        tensor lives in (the unused `nextn` tensors at the end of GLM-4.5-Air
        are one), so a floor chunk can straddle the boundary. The first live
        run copied nothing for such a chunk, marked it done, and the client
        faulted on the same page five million times. Now the copy bisects to
        the mapped prefix and copies that.
        """
        if m.holes and m.in_hole(a, b):
            # a known hole: copy around it rather than discovering it by ENOENT
            skipped = 0
            for x, y in m.live_pieces(a, b):
                skipped += self._copy_pages(m, x, y, source)
            return skipped
        data = source(a, b)
        # The bytes object's own buffer is the copy source: no second memcpy
        # of a 9 MB slab in Python. ``data`` stays referenced until the loop
        # ends, which is what keeps the address valid.
        src0 = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
        cur = a
        end = b            # what this attempt asks for; shrinks on ENOENT, resets on progress
        t0 = time.perf_counter()
        skipped = 0
        while cur < b:
            req = bytearray(struct.pack("<QQQQq", m.base + cur, src0 + (cur - a), end - cur, 0, 0))
            try:
                fcntl.ioctl(m.uffd, UFFDIO_COPY, req)
                done = struct.unpack_from("<q", req, 32)[0]
                cur += done if done > 0 else (end - cur)
                end = b
            except OSError as e:
                done = struct.unpack_from("<q", req, 32)[0]
                if done > 0:
                    cur += done
                    end = b
                elif e.errno == errno.EEXIST:
                    cur += PAGE
                    skipped += 1
                    end = b
                elif e.errno == errno.EAGAIN:
                    # the client's address space is changing under us (an
                    # mmap/munmap in flight); back off briefly and retry
                    self.stats.copy_eagain += 1
                    if self.stats.copy_eagain % 1000 == 0:
                        self._say(f"UFFDIO_COPY keeps returning EAGAIN ({self.stats.copy_eagain} so far)")
                    time.sleep(0.001)
                elif e.errno == errno.ENOENT:
                    # part of [cur, end) is unmapped: halve the ask and try again.
                    # A single page that is gone is skipped, not treated as the
                    # end — llama.cpp unmaps a *prefix* fragment too (whatever
                    # precedes the first used tensor), and the first live run
                    # stopped at it and left the rest of the chunk unserved.
                    if end - cur <= PAGE:
                        self.stats.unmapped_pages += 1
                        cur += PAGE
                        end = b
                    else:
                        end = cur + max(PAGE, ((end - cur) // 2) // PAGE * PAGE)
                elif e.errno == errno.ESRCH:
                    # the client process is gone; nothing to serve any more
                    if not m.dead:
                        m.dead = True
                        self._say(f"pid {m.pid} has exited; its mapping of {m.path.name} is retired")
                    return skipped
                else:
                    raise LoaderError(f"UFFDIO_COPY at {m.base + cur:#x}: {e}") from e
        self.stats.copy_seconds += time.perf_counter() - t0
        self.stats.bytes_copied += b - a
        if self.debug:
            self._say(f"copied [{a}, {b}) {b - a} bytes, {skipped} pages already present, "
                      f"{(time.perf_counter() - t0) * 1000:.1f} ms")
        return skipped

    # -- eviction -------------------------------------------------------------

    def _evicted(self, key) -> None:
        """The cache chose a victim: drop its interior pages in the client."""
        if isinstance(key, tuple) and key and key[0] == "floor":
            return
        self.stats.evictions += 1
        if key in self._prefetched:
            self._prefetched.discard(key)
            self.stats.prefetch_wasted += 1
        for m in self.mappings:
            regions = m.layout.by_key.get(key)
            if not regions or m.dead:
                continue
            conn = self._evict_conns.get(m.pid)
            if conn is None:
                continue
            for r in regions:
                a = (r.start + PAGE - 1) & ~(PAGE - 1)      # interior pages only: the edge
                b = r.end & ~(PAGE - 1)                      # pages are shared with a neighbour
                for x, y in m.live_pieces(a, b) if m.holes else [(a, b)]:
                    if y <= x:
                        continue
                    try:
                        conn.sendall(f"EVICT {m.base + x:x} {y - x}\n".encode())
                        self.stats.evict_bytes += y - x
                    except OSError as e:
                        self._say(f"eviction channel to pid {m.pid} lost: {e}")

    # -- the client unmapped part of a served region ---------------------------

    def _on_unmap(self, pid: int, line: str) -> None:
        # UNMAP <addr> <len>: llama.cpp dropped a fragment (prefix before the
        # first CPU tensor, suffix after the last). Whatever lived there is
        # gone from the client, must never be evicted into (the addresses can
        # be anyone's now) and cannot fault again.
        _, addr_s, len_s = line.split()
        addr, n = int(addr_s, 16), int(len_s)
        m = next((x for x in self.mappings if x.pid == pid and x.contains(addr)), None)
        if m is None:
            self._say(f"pid {pid}: UNMAP {addr:#x}+{n} is not inside a mapping I serve")
            return
        # munmap takes whole pages: round the way the kernel does. The region
        # is page-granular; the file (m.length) is not.
        a = (addr - m.base) & ~(PAGE - 1)
        b = min((addr - m.base + n + PAGE - 1) & ~(PAGE - 1), (m.length + PAGE - 1) & ~(PAGE - 1))
        if a == 0 and b >= m.length:
            # the whole region: the client is tearing down (Python's mmap
            # object, llama_free_model). Nothing to forget one by one; the
            # mapping is retired and nothing is evicted into it again.
            with self._lock:
                m.holes.append((a, b))
                m.dead = True
                self.stats.unmaps += 1
                self.stats.unmapped_bytes += b - a
            self._say(f"pid {pid}: unmapped all of {m.path.name}; mapping retired")
            return
        forgotten = 0
        with self._lock:
            m.holes.append((a, b))
            self.stats.unmaps += 1
            self.stats.unmapped_bytes += b - a
            lo = max(0, bisect.bisect_right(m.layout._starts, a) - 1)   # the region a falls in, too
            seen = set()
            for r in m.layout.regions[lo:]:
                if r.start >= b:
                    break
                key = r.key
                if key in seen or key[0] == "floor":
                    continue
                seen.add(key)
                # gone when any of its slabs lost an interior page: it cannot be
                # served whole again, and its bytes must not be evicted into
                if any(m.in_hole((x.start + PAGE - 1) & ~(PAGE - 1), x.end & ~(PAGE - 1))
                       for x in m.layout.by_key[key]):
                    m.gone.add(key)
                    if self.cache.forget(key):
                        forgotten += 1
                        self._prefetched.discard(key)
            self.stats.forgotten += forgotten
        self._say(f"pid {pid}: unmapped [{a}, {b}) of {m.path.name} ({(b - a) / GB:.2f} GB); "
                  f"{len(m.gone)} experts gone, {forgotten} of them were resident")

    # -- routing in -----------------------------------------------------------

    def _on_route(self, line: str, *, after: bool = False) -> None:
        # ROUTE <layer> <n_tokens> <n_used> e,e;e,e   — before the layer runs (llama.cpp's cb_eval)
        # ROUTED …                                   — after the step ran (a CUDA-graph runtime
        #                                              reads the ids back afterwards); a hit is
        #                                              then "was not faulted in during this token"
        head, _, body = line.partition(" ")
        parts = line.split(" ", 4)
        layer, n_tokens, n_used = int(parts[1]), int(parts[2]), int(parts[3])
        rows = [[int(x) for x in row.split(",") if x] for row in parts[4].split(";")] if len(parts) > 4 else []
        if layer <= self._last_layer or (after and self._last_layer < 0):
            self._end_token()               # a ROUTED burst always starts at a boundary, the first one too
        self._last_layer = layer
        keys = {(layer, e) for row in rows for e in row}
        with self._lock:
            for key in keys:
                self.stats.routed += 1
                if after:
                    if key in self._after_faulted:
                        self.stats.misses += 1
                        self.cache.stats.misses += 1
                    else:
                        self.stats.hits += 1
                        if key in self._prefetched:
                            self._prefetched.discard(key)
                            self.stats.prefetch_useful += 1
                    if key in self.cache:
                        self.cache.get(key)
                    continue
                if key in self.cache:
                    self.stats.hits += 1
                    self.cache.get(key)
                    if key in self._prefetched:
                        self._prefetched.discard(key)
                        self.stats.prefetch_useful += 1
                elif key in self._serving:
                    self.stats.misses += 1
                    self.stats.prefetch_late += 1
                else:
                    self.stats.misses += 1
                    self.cache.stats.misses += 1
        self._token_keys |= keys
        if rows:
            self._sofar[layer] = rows[-1]
        if self.depth > 0:
            if after:
                # the step already ran: guesses for the *next* step, once the
                # burst has named every served layer
                self.scheduler.routed(layer, self.stats.tokens)
                if layer == self._served_layers()[-1]:
                    self._prefetch_next_token()
            else:
                self._prefetch_after(layer)

    def _prefetch_after(self, layer: int) -> None:
        """Queue the predictor's guesses for the layers ahead, by class.

        The next layer's top guesses are HIGH_CONFIDENCE_NEXT; the layer after
        that, and the lower-ranked half of the next layer, PROBABLE_NEXT. Hot
        experts by frequency go in as BACKGROUND_HOT when the tier has room
        for them without evicting anything — a guess that evicts a resident
        expert to make room for a "hot" one is a guess LRU already made.
        """
        moe = self.source.moe_layers
        token = self.stats.tokens
        self.scheduler.routed(layer, token)
        ahead = [l for l in moe if l > layer][:2]
        for i, nxt in enumerate(ahead):
            pred = self.predictor.predict(nxt, dict(self._sofar))
            top = pred.top(self.depth)
            for rank, e in enumerate(top):
                key = (nxt, e)
                with self._lock:
                    if key in self.cache or key in self._serving:
                        continue
                if not any(key in m.layout.by_key for m in self.mappings):
                    continue
                prio = (Priority.HIGH_CONFIDENCE_NEXT if i == 0 and rank < max(1, self.depth // 2)
                        else Priority.PROBABLE_NEXT)
                if self.scheduler.submit(key, layer=nxt, token=token, priority=prio):
                    self.stats.prefetch_issued += 1
        if self.cache.free_bytes > 4 * self.source.expert_nbytes_max() and self.tracker.tokens_seen > 8:
            for key in self.tracker.hot(4):
                with self._lock:
                    if key in self.cache or key in self._serving:
                        continue
                if self.scheduler.submit(key, layer=key[0], token=token, priority=Priority.BACKGROUND_HOT):
                    self.stats.prefetch_issued += 1

    def _served_layers(self) -> list[int]:
        """The MoE layers some client has a mapping for (FreeToken: only its tiered layers)."""
        n = len(self.mappings)
        if getattr(self, "_served_cache", (None, None))[0] != n:
            layers = sorted({k[0] for m in self.mappings for k in m.layout.by_key if k[0] != "floor"})
            self._served_cache = (n, layers or [-1])
        return self._served_cache[1]

    def _prefetch_next_token(self) -> None:
        """After a ROUTED burst: for every served layer, the predictor's guesses
        for the next step (frequency, persistence; there is no within-token
        context across steps), as PROBABLE_NEXT."""
        token = self.stats.tokens
        for layer in self._served_layers():
            if layer < 0:
                continue
            pred = self.predictor.predict(layer, {})
            for e in pred.top(self.depth):
                key = (layer, e)
                with self._lock:
                    if key in self.cache or key in self._serving:
                        continue
                if not any(key in m.layout.by_key for m in self.mappings):
                    continue
                # for the next step: a job for a layer this step already routed would be stale
                if self.scheduler.submit(key, layer=layer, token=token + 1, priority=Priority.PROBABLE_NEXT):
                    self.stats.prefetch_issued += 1

    def _serve_batch(self, keys: list) -> None:
        """Materialise several predicted experts of one layer, reading adjacent
        slabs together: experts e and e+1 are neighbouring slabs in each fused
        tensor, so a run of consecutive ids is one pread per projection."""
        keys = sorted(k for k in keys if isinstance(k, tuple) and k[0] != "floor")
        if not keys:
            return
        m = next((x for x in self.mappings if keys[0] in x.layout.by_key), None)
        if m is None:
            return
        keys = [k for k in keys if k not in m.gone]      # never into memory the client unmapped
        # split into runs of consecutive expert ids
        runs: list[list] = []
        for k in keys:
            if runs and runs[-1][-1][0] == k[0] and runs[-1][-1][1] + 1 == k[1]:
                runs[-1].append(k)
            else:
                runs.append([k])
        for run in runs:
            if len(run) == 1:
                self._serve_expert(m, run[0], why="prefetch")
                continue
            self._serve_run(m, run)

    def _serve_run(self, m: Mapping, run: list) -> None:
        """A run of consecutive experts: one read per projection for the whole run."""
        with self._lock:
            todo = [k for k in run if k not in self.cache and k not in self._serving]
            evs = {}
            for k in todo:
                evs[k] = self._serving[k] = threading.Event()
        if not todo:
            return
        try:
            # every mapping of this client holding the run (one file for a GGUF;
            # one buffer per bank for an FTW model), one read per projection each
            for mm in self._mappings_with(todo[0], m.pid):
                first, last = mm.layout.by_key[todo[0]], mm.layout.by_key[todo[-1]]
                # regions of an expert are in projection order; pair them up
                for proj in range(len(first)):
                    a = first[proj].start & ~(PAGE - 1)
                    b = min((last[proj].end + PAGE - 1) & ~(PAGE - 1), (mm.length + PAGE - 1) & ~(PAGE - 1))
                    data = self._file_bytes(mm, a, b)          # one read for the whole run
                    self.stats.coalesced_reads += 1
                    self._copy_pages(mm, a, b, lambda x, y, d=data, a0=a: d[x - a0:y - a0])
            with self._lock:
                for k in todo:
                    self.cache.put(k, None, sum(r.end - r.start for mm in self._mappings_with(k, m.pid)
                                                for r in mm.layout.by_key[k]))
                    self._prefetched.add(k)
                    self.stats.coalesced_experts += 1
        finally:
            with self._lock:
                for k in todo:
                    self._serving.pop(k, None)
            for ev in evs.values():
                ev.set()

    def _end_token(self) -> None:
        with self._lock:
            # what a ROUTED burst arriving now is scored against: the faults of the
            # step that just ran, which the burst describes (its first line is the
            # boundary; clearing before scoring made every expert a hit)
            self._after_faulted = self._token_faulted
            self._token_faulted = set()
        if self._token_keys:
            self.tracker.record(self._token_keys)
            self.tracker.begin_token()
            routing = defaultdict(list)
            for l, e in self._token_keys:
                routing[l].append(e)
            self.predictor.observe(dict(routing))
            self.stats.tokens += 1
            if self.tel:
                self.tel.event("token", **self._token_delta())
            if self.verbose and (self.stats.tokens <= 3 or self.stats.tokens % 20 == 0):
                s = self.stats
                self._say(f"token {s.tokens}: hit {s.hit_rate:.1%} ({s.hits}/{s.hits + s.misses}) "
                          f"faults {s.faults} copied {s.bytes_copied / GB:.1f} GB "
                          f"evicted {s.evictions} prefetch {s.prefetch_issued}/{s.prefetch_useful}/"
                          f"{s.prefetch_late}/{s.prefetch_wasted} resident {len(self.cache)} "
                          f"({self.cache.used_bytes / GB:.1f} GB)")
            if self.tel and self.stats.tokens % 10 == 0:
                self.tel.snapshot(values=self._snapshot_values())
        self._token_keys = set()
        self._sofar = {}
        now = time.perf_counter()
        self._token_started = now

    def _token_delta(self) -> dict:
        s = self.stats
        cur = {"loader.hits": s.hits, "loader.misses": s.misses, "loader.faults": s.faults,
               "loader.bytes_copied": s.bytes_copied, "loader.evictions": s.evictions,
               "loader.prefetch_issued": s.prefetch_issued, "loader.prefetch_useful": s.prefetch_useful,
               "loader.prefetch_late": s.prefetch_late, "loader.prefetch_wasted": s.prefetch_wasted}
        prev = self._token_stats or {k: 0 for k in cur}
        self._token_stats = cur
        out = {k.replace("loader.", ""): v - prev[k] for k, v in cur.items()}
        out["wall_ms"] = (time.perf_counter() - self._token_started) * 1000
        out["resident"] = len(self.cache)
        out["resident_bytes"] = self.cache.used_bytes
        return out

    def _snapshot_values(self) -> dict:
        s = self.stats
        return {f"loader.{k}": getattr(s, k) for k in
                ("faults", "faults_expert", "faults_floor", "faults_duplicate", "faults_resident",
                 "faults_repaired", "bytes_copied",
                 "copy_seconds", "read_seconds", "routed", "hits", "misses", "prefetch_issued",
                 "prefetch_useful", "prefetch_late", "prefetch_wasted", "evictions", "evict_bytes",
                 "tokens", "read_retries", "coalesced_reads", "coalesced_experts",
                 "unmaps", "unmapped_bytes", "forgotten", "unmapped_pages",
                 "repeat_faults", "copy_eagain")} | {
            "loader.prefetch_enqueued": self.scheduler.enqueued, "loader.prefetch_served": self.scheduler.served,
            "loader.prefetch_superseded": self.scheduler.superseded,
            "loader.prefetch_dropped_full": self.scheduler.dropped_full,
            **{f"loader.prefetch_class_{Priority.NAMES[p]}": n for p, n in self.scheduler.by_class.items()},
            "loader.resident": len(self.cache), "loader.resident_bytes": self.cache.used_bytes,
            "storage.bytes_read": self.backend.stats.bytes_read,
            "storage.operations": self.backend.stats.operations,
            "storage.seconds": self.backend.stats.seconds}

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"tierinfer-loader: {msg}", file=sys.stderr, flush=True)


# -- wire helpers -----------------------------------------------------------


def _readline(conn: socket.socket) -> str:
    out = bytearray()
    while True:
        c = conn.recv(1)
        if not c:
            return out.decode(errors="replace")
        if c == b"\n":
            return out.decode(errors="replace")
        out += c


class _LineReader:
    """Lines off a stream socket, with any SCM_RIGHTS descriptors that arrived
    while a line was being assembled handed over with that line."""

    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn
        self.buf = bytearray()
        self.fds: list[int] = []

    def line(self) -> tuple[str | None, list[int]]:
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = self.buf[:i].decode(errors="replace")
                del self.buf[:i + 1]
                fds, self.fds = self.fds, []
                return line, fds
            data, anc, _, _ = self.conn.recvmsg(65536, socket.CMSG_SPACE(4 * 4))
            for level, typ, cdata in anc:
                if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                    self.fds.extend(array.array("i", cdata[:len(cdata) - len(cdata) % 4]))
            if not data:
                if self.buf:
                    line, self.buf = self.buf.decode(errors="replace"), bytearray()
                    fds, self.fds = self.fds, []
                    return line, fds
                return None, self.fds
            self.buf += data

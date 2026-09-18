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
import fcntl
import mmap
import os
import select
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
UFFD_EVENT_PAGEFAULT = 0x12
_MSG = 32                                          # sizeof(struct uffd_msg)

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class LoaderError(RuntimeError):
    pass


# -- the layout of a mapping ------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A byte range of one file and what it is: an expert's slab or floor."""

    start: int
    end: int
    key: object          # (layer, expert) for a routed expert; ("floor", tensor) otherwise
    name: str


class FileLayout:
    """Every region of one model file, addressable by offset in O(log n)."""

    def __init__(self, index: ModelIndex, path: Path) -> None:
        self.path = path
        self.size = path.stat().st_size
        regions: list[Region] = []
        expert_tensors = set()
        for layer in index.moe_layers:
            for e in range(index.expert_count):
                ref = index.expert(layer, e)
                for r in ref.ranges:
                    if r.path == path:
                        regions.append(Region(r.file_offset, r.end, (layer, e), r.name))
                        expert_tensors.add(r.name.split("#")[0])
        for t in index.gguf.tensors:
            if t.path == path and t.name not in expert_tensors:
                regions.append(Region(t.file_offset, t.file_offset + t.nbytes, ("floor", t.name), t.name))
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

    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.base + self.length


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

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class LoaderServer:
    """Accepts mappings from preloaded llama.cpp processes and serves them."""

    def __init__(self, index: ModelIndex, *, ram_bytes: int, workers: int = 8, depth: int = 0,
                 telemetry: Telemetry | None = None, floor_chunk: int = 16 * MB,
                 predictor: Predictor | None = None, verbose: bool = True,
                 drop_page_cache: bool = True) -> None:
        self.index = index
        self.drop_page_cache = drop_page_cache
        self.ram_bytes = ram_bytes
        self.workers = workers
        self.depth = depth
        self.floor_chunk = floor_chunk
        self.verbose = verbose
        self.tel = telemetry
        self.backend = StorageBackend.for_model(index.gguf)
        self.layouts: dict[Path, FileLayout] = {}
        self.mappings: list[Mapping] = []
        self.tracker = ExpertTracker(window=128)
        self.predictor = predictor or AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16)
        self.cache = _ResidencyCache(ram_bytes, self.tracker, self._evicted)
        self.stats = LoaderStats()
        self._evict_conns: dict[int, socket.socket] = {}       # pid -> evict socket
        self._lock = threading.RLock()
        self._serving: dict[object, threading.Event] = {}      # key -> done event
        self._prefetched: set = set()                          # keys materialised by prefetch, not yet routed
        self._floor_done: set = set()                          # (path, chunk index)
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tierinfer-fault")
        self._prefetch_pool = ThreadPoolExecutor(max_workers=max(1, workers // 2),
                                                 thread_name_prefix="tierinfer-prefetch")
        # routing state for the current token
        self._last_layer = -1
        self._sofar: dict[int, list[int]] = {}
        self._token_keys: set = set()
        self._token_started = time.perf_counter()
        self._token_stats = None

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
                  f"({self.ram_bytes // max(1, self.index.expert_nbytes_max())} experts), "
                  f"{self.workers} fault workers, prefetch depth {self.depth}")
        if self.tel:
            self.tel.open_run(model=str(self.index.gguf.path), ram_bytes=self.ram_bytes,
                              workers=self.workers, depth=self.depth, files=len(self.index.gguf.files))
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
        self._prefetch_pool.shutdown(wait=False)
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
        finally:
            self._say(f"pid {pid}: control channel closed")
            f.close()

    def _on_map(self, conn: socket.socket, pid: int, line: str, fds: list[int]) -> None:
        _, path_s, base_s, len_s = line.split()
        path = Path(path_s)
        if not fds:
            conn.sendall(b"NO no descriptor attached\n")
            return
        uffd = fds[0]
        if path not in {Path(p) for p in self.index.gguf.files}:
            conn.sendall(b"NO not a file of this model\n")
            os.close(uffd)
            return
        layout = self.layouts.get(path)
        if layout is None:
            layout = self.layouts[path] = FileLayout(self.index, path)
        m = Mapping(path=path, base=int(base_s, 16), length=int(len_s), uffd=uffd, layout=layout, pid=pid)
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
                    continue
                _flags, addr = struct.unpack_from("<QQ", data, off + 8)
                self._fault(addr)

    def _fault(self, addr: int) -> None:
        m = self._mapping_for(addr)
        if m is None:
            self._say(f"fault at {addr:#x} outside every mapping — cannot serve")
            return
        offset = (addr - m.base) & ~(PAGE - 1)
        region = m.layout.owner_of_page(offset)
        self.stats.faults += 1
        if region is None:
            # Padding between tensors, or the tail past EOF: the file's bytes
            # there (or zeros past its end) — never anything else.
            self._copy_pages(m, offset, offset + PAGE, lambda a, b: self._file_bytes(m, a, b))
            return
        if region.key[0] == "floor":
            self.stats.faults_floor += 1
            self._serve_floor(m, offset)
        else:
            self.stats.faults_expert += 1
            self._serve_expert(m, region.key, why="fault")

    def _mapping_for(self, addr: int) -> Mapping | None:
        for m in self.mappings:
            if m.contains(addr):
                return m
        return None

    # -- serving ------------------------------------------------------------

    def _serve_expert(self, m: Mapping, key: object, *, why: str) -> None:
        """Materialise all slabs of one expert in the client, once."""
        with self._lock:
            if key in self.cache:
                if why == "fault":
                    # resident by our books, yet it faulted: an edge page the
                    # eviction of a neighbour took, or a race with eviction.
                    pass
                else:
                    return
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
            regions = m.layout.by_key[key]
            layer, expert = key
            nbytes = 0
            for r in regions:
                a = r.start & ~(PAGE - 1)
                b = min((r.end + PAGE - 1) & ~(PAGE - 1), (m.length + PAGE - 1) & ~(PAGE - 1))
                self._copy_pages(m, a, b, lambda x, y, r=r: self._file_bytes(m, x, y))
                nbytes += r.end - r.start
            with self._lock:
                admitted = self.cache.put(key, None, nbytes)
                if why == "prefetch":
                    self._prefetched.add(key)
                if not admitted:
                    self._say(f"expert {key} does not fit the tier at all ({nbytes} bytes)")
        finally:
            with self._lock:
                self._serving.pop(key, None)
            ev.set()

    def _serve_floor(self, m: Mapping, offset: int) -> None:
        """Materialise the aligned chunk of floor around ``offset``, pinned."""
        chunk = offset // self.floor_chunk
        tag = (m.path, chunk)
        with self._lock:
            if tag in self._floor_done:
                return
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
        """The file's bytes for [a, b), zero-padded past EOF. Exact path, retried once."""
        end = min(b, m.layout.size)
        want = ByteRange(name=f"{m.path.name}:{a}", file_offset=a, nbytes=end - a, path=m.path)
        t0 = time.perf_counter()
        try:
            blobs, _ = self.backend.read([want])
        except OSError as e:
            self.stats.read_retries += 1
            self._say(f"read of {want.name} failed ({e}); retrying once")
            blobs, _ = self.backend.read([want])
        self.stats.read_seconds += time.perf_counter() - t0
        if self.drop_page_cache:
            # The bytes now live in the client's mapping; a second copy in the
            # page cache would be RAM the tier does not account for, and would
            # make an eviction a lie (the next fault would be served from RAM).
            try:
                self.backend.evict([want])
            except OSError as e:
                self._say(f"could not drop page cache behind {want.name}: {e}")
        data = blobs[0]
        if end < b:
            data += bytes(b - end)
        return data

    def _copy_pages(self, m: Mapping, a: int, b: int, source: Callable[[int, int], bytes]) -> None:
        """UFFDIO_COPY the file's bytes for pages [a, b) of the mapping.

        Pages already present (a neighbour's edge, a race with another
        worker) return EEXIST; they are skipped a page at a time, because a
        present page is by construction already the right bytes.
        """
        data = source(a, b)
        buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
        src0 = ctypes.addressof(buf)
        cur = a
        t0 = time.perf_counter()
        while cur < b:
            req = bytearray(struct.pack("<QQQQq", m.base + cur, src0 + (cur - a), b - cur, 0, 0))
            try:
                fcntl.ioctl(m.uffd, UFFDIO_COPY, req)
                done = struct.unpack_from("<q", req, 32)[0]
                cur += done if done > 0 else (b - cur)
            except OSError as e:
                done = struct.unpack_from("<q", req, 32)[0]
                if done > 0:
                    cur += done
                elif e.errno == errno.EEXIST:
                    cur += PAGE
                elif e.errno == errno.EAGAIN:
                    # the client's address space is changing under us (an
                    # mmap/munmap in flight); back off briefly and retry
                    self.stats.copy_eagain += 1
                    if self.stats.copy_eagain % 1000 == 0:
                        self._say(f"UFFDIO_COPY keeps returning EAGAIN ({self.stats.copy_eagain} so far)")
                    time.sleep(0.001)
                elif e.errno == errno.ENOENT:
                    # the client unmapped this range (llama.cpp frees fragments
                    # it moved to the GPU); nothing left to serve here
                    return
                else:
                    raise LoaderError(f"UFFDIO_COPY at {m.base + cur:#x}: {e}") from e
        self.stats.copy_seconds += time.perf_counter() - t0
        self.stats.bytes_copied += b - a

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
            if not regions:
                continue
            conn = self._evict_conns.get(m.pid)
            if conn is None:
                continue
            for r in regions:
                a = (r.start + PAGE - 1) & ~(PAGE - 1)      # interior pages only: the edge
                b = r.end & ~(PAGE - 1)                      # pages are shared with a neighbour
                if b > a:
                    try:
                        conn.sendall(f"EVICT {m.base + a:x} {b - a}\n".encode())
                        self.stats.evict_bytes += b - a
                    except OSError as e:
                        self._say(f"eviction channel to pid {m.pid} lost: {e}")
            break

    # -- routing in -----------------------------------------------------------

    def _on_route(self, line: str) -> None:
        # ROUTE <layer> <n_tokens> <n_used> e,e;e,e
        head, _, body = line.partition(" ")
        parts = line.split(" ", 4)
        layer, n_tokens, n_used = int(parts[1]), int(parts[2]), int(parts[3])
        rows = [[int(x) for x in row.split(",") if x] for row in parts[4].split(";")] if len(parts) > 4 else []
        if layer <= self._last_layer:
            self._end_token()
        self._last_layer = layer
        keys = {(layer, e) for row in rows for e in row}
        with self._lock:
            for key in keys:
                self.stats.routed += 1
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
            self._prefetch_after(layer)

    def _prefetch_after(self, layer: int) -> None:
        nxt = layer + 1
        if nxt not in set(self.index.moe_layers):
            return
        pred = self.predictor.predict(nxt, dict(self._sofar))
        m = self.mappings[0] if self.mappings else None
        if m is None:
            return
        for e in pred.top(self.depth):
            key = (nxt, e)
            with self._lock:
                if key in self.cache or key in self._serving:
                    continue
            # the expert's slabs live in whichever mapping holds that layer
            target = next((x for x in self.mappings if key in x.layout.by_key), None)
            if target is None:
                continue
            self.stats.prefetch_issued += 1
            self._prefetch_pool.submit(self._serve_expert, target, key, why="prefetch")

    def _end_token(self) -> None:
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
                ("faults", "faults_expert", "faults_floor", "faults_duplicate", "bytes_copied",
                 "copy_seconds", "read_seconds", "routed", "hits", "misses", "prefetch_issued",
                 "prefetch_useful", "prefetch_late", "prefetch_wasted", "evictions", "evict_bytes",
                 "tokens", "read_retries")} | {
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

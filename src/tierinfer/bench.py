"""Measurement primitives for a reproducible baseline, none of them needing root.

The baseline experiment asks one question: what does it cost to run a model
that does not fit in the memory you are willing to give it? Answering it
honestly needs four things that are easy to get wrong.

**A cold cache.** A 56 GB model on a 187 GB host is resident after the first
run, so every later run measures RAM, not NVMe. ``posix_fadvise(DONTNEED)``
drops the file's page cache without root and without disturbing anything else
on the machine.

**A memory limit.** The interesting condition is a host smaller than the
model, which this host is not. ``systemd-run --user --scope -p MemoryMax=``
supplies one, because the ``memory`` controller is delegated to the user
slice here. The scope also reports ``memory.peak``, which is what the run
actually took rather than what it asked for.

**Device counters.** ``/proc/diskstats`` gives reads, sectors and service
time per device, so bandwidth, IOPS, mean read size and await come out of a
before/after difference. The partition is preferred over the whole disk so
that traffic to other filesystems does not land in the measurement.

**Residency.** ``mincore`` reports which pages of the model are actually in
core, which is the only way to tell a run that streamed 56 GB from one that
streamed 6 GB and reused it.

What this cannot do without root: a true read-size *histogram*. ``blktrace``
and the block tracepoints need privileges. The mean read size derived from
sectors over reads is reported instead, and it is labelled a mean, because a
mean of 132 KB is consistent with both a uniform 132 KB stream and a mix of
4 KB faults and 2 MB readahead — and those are different worlds.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path

PAGE = mmap.PAGESIZE
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


class BenchError(RuntimeError):
    pass


# -- device counters ----------------------------------------------------


@dataclass(frozen=True)
class DiskCounters:
    """Raw /proc/diskstats read fields for one block device."""

    device: str
    reads: int
    sectors: int
    ms: int
    at: float

    def __sub__(self, other: "DiskCounters") -> "DiskDelta":
        if self.device != other.device:
            raise BenchError(f"counters from different devices: {self.device} vs {other.device}")
        return DiskDelta(
            device=self.device,
            reads=self.reads - other.reads,
            bytes_read=(self.sectors - other.sectors) * 512,
            ms=self.ms - other.ms,
            seconds=self.at - other.at,
        )


@dataclass(frozen=True)
class DiskDelta:
    device: str
    reads: int
    bytes_read: int
    ms: int
    seconds: float

    @property
    def gb_read(self) -> float:
        return self.bytes_read / 1024 ** 3

    @property
    def bandwidth_gbps(self) -> float:
        return self.gb_read / self.seconds if self.seconds > 0 else 0.0

    @property
    def iops(self) -> float:
        return self.reads / self.seconds if self.seconds > 0 else 0.0

    @property
    def mean_read_bytes(self) -> float:
        """Mean, not a distribution — see the module docstring."""
        return self.bytes_read / self.reads if self.reads else 0.0

    @property
    def await_ms(self) -> float:
        return self.ms / self.reads if self.reads else 0.0


def device_for(path: str | os.PathLike) -> str:
    """The block device backing this path, preferring the partition.

    Traffic to a sibling partition is not this model's traffic, so the
    narrower device is the better measurement even though the whole-disk row
    is the more familiar one.
    """
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    link = Path(f"/sys/dev/block/{major}:{minor}")
    if not link.exists():
        raise BenchError(f"no block device for {path} (dev {major}:{minor})")
    name = link.resolve().name
    if name not in _diskstat_rows():
        raise BenchError(f"{name} has no /proc/diskstats row")
    return name


def _diskstat_rows() -> dict[str, list[str]]:
    rows = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) >= 14:
            rows[f[2]] = f
    return rows


def disk_counters(device: str) -> DiskCounters:
    row = _diskstat_rows().get(device)
    if row is None:
        raise BenchError(f"no /proc/diskstats row for {device}")
    return DiskCounters(device=device, reads=int(row[3]), sectors=int(row[5]),
                        ms=int(row[6]), at=time.monotonic())


# -- page cache ---------------------------------------------------------


@dataclass(frozen=True)
class Residency:
    path: str
    resident_pages: int
    total_pages: int

    @property
    def resident_bytes(self) -> int:
        return self.resident_pages * PAGE

    @property
    def fraction(self) -> float:
        return self.resident_pages / self.total_pages if self.total_pages else 0.0


def residency(path: str | os.PathLike, span: int | None = None) -> Residency:
    """How much of this file is in core, via mincore.

    The map is private and writable so that its address can be taken through
    the buffer protocol; nothing is ever written, so no page is ever copied
    and mincore reports the file's own page cache residency.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        length = size if span is None else min(span, size)
        if length == 0:
            return Residency(str(path), 0, 0)
        mm = mmap.mmap(fd, length, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            pages = (length + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * pages)()
            if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(length), vec) != 0:
                raise BenchError(f"mincore failed: {os.strerror(ctypes.get_errno())}")
            resident = sum(1 for b in vec if b & 1)
            del vec
            return Residency(str(path), resident, pages)
        finally:
            mm.close()
    finally:
        os.close(fd)


def drop_cache(path: str | os.PathLike) -> Residency:
    """Evict this file from the page cache and prove it, without root.

    The file is flushed first: ``DONTNEED`` skips dirty pages, so a model
    that was just written or copied would otherwise stay wholly resident
    while reporting a successful eviction.

    Returns the residency measured *after* the eviction. It is not always
    zero: a page another process still has mapped will not go, and saying so
    is more useful than asserting a clean slate.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        # A dirty page is not dropped by DONTNEED — it has to reach the disk
        # first. Without this, a file that was just written or copied stays
        # fully resident and every "cold" run after it silently measures RAM.
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return residency(path)


def warm_cache(path: str | os.PathLike, chunk: int = 32 << 20) -> Residency:
    """Read the whole file so the next run measures RAM rather than NVMe."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        while os.read(fd, chunk):
            pass
    finally:
        os.close(fd)
    return residency(path)


# -- running a command under a memory limit -----------------------------


@dataclass
class Execution:
    argv: list[str]
    exit_code: int
    wall_seconds: float
    stdout: str
    memory_max_bytes: int | None = None
    peak_memory_bytes: int | None = None
    timed_out: bool = False


def memory_controller_available() -> bool:
    """Whether this user may cap memory, i.e. whether the controller is delegated."""
    if shutil.which("systemd-run") is None:
        return False
    for p in Path("/sys/fs/cgroup").rglob("user@*.service/cgroup.controllers"):
        try:
            if "memory" in p.read_text().split():
                return True
        except OSError:
            continue
    return False


def run_limited(argv: list[str], *, memory_max_bytes: int | None = None,
                timeout: float = 3600.0, unit: str | None = None) -> Execution:
    """Run argv, optionally inside a user scope with a memory ceiling.

    Without a limit this is a plain subprocess; the scope is only introduced
    when it is asked for, so the unconstrained conditions are not measured
    through a layer the constrained one adds.

    The peak is sampled from the scope's own ``memory.peak`` while the child
    runs, because systemd removes a transient scope as soon as it exits and
    asking afterwards usually returns nothing. ``memory.peak`` is a high
    watermark, so any read before teardown carries the whole run.
    """
    unit = unit or f"tierinfer-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    if memory_max_bytes is not None:
        if not memory_controller_available():
            raise BenchError("memory controller is not delegated to this user; "
                             "the constrained condition cannot be measured here")
        cmd = ["systemd-run", "--user", "--scope", "--quiet",
               f"--unit={unit}",
               f"--property=MemoryMax={memory_max_bytes}",
               "--property=MemorySwapMax=0", *argv]
    else:
        cmd = list(argv)

    peak = _PeakSampler(unit) if memory_max_bytes is not None else None
    start = time.monotonic()
    timed_out = False
    try:
        if peak:
            peak.start()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
        code = -1
    finally:
        if peak:
            peak.stop()
    wall = time.monotonic() - start

    peak_bytes = None
    if peak:
        peak_bytes = peak.value or _scope_peak(unit)
    return Execution(argv=list(argv), exit_code=code, wall_seconds=wall, stdout=out,
                     memory_max_bytes=memory_max_bytes, peak_memory_bytes=peak_bytes,
                     timed_out=timed_out)


def _scope_cgroup(unit: str) -> Path | None:
    """The cgroup directory of a transient user scope, if it exists yet."""
    root = Path("/sys/fs/cgroup/user.slice")
    try:
        for p in root.glob(f"**/{unit}.scope"):
            return p
    except OSError:
        pass
    return None


class _PeakSampler:
    """Reads a scope's memory.peak until the scope goes away.

    The glob that finds the cgroup is the expensive part, so it runs until
    it succeeds and the path is then kept: every later sample is one small
    file read. That makes a 10 ms interval cheap enough to also catch a run
    that lasts a fraction of a second — a scope lives only as long as its
    command, and a sampler that first looks 200 ms in reports nothing at all
    for a quick one.
    """

    def __init__(self, unit: str, interval: float = 0.01) -> None:
        self.unit = unit
        self.interval = interval
        self.value: int | None = None
        self._path: Path | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _read_once(self) -> None:
        if self._path is None:
            cg = _scope_cgroup(self.unit)
            if cg is None:
                return
            for name in ("memory.peak", "memory.current"):
                if (cg / name).exists():
                    self._path = cg / name
                    break
            else:
                return
        try:
            v = int(self._path.read_text().strip())
        except (OSError, ValueError):
            return
        if self.value is None or v > self.value:
            self.value = v

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read_once()
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self._read_once()          # one last look before systemd tears it down
        if self._thread:
            self._thread.join(timeout=2.0)


# -- one measured condition ---------------------------------------------


@dataclass
class Measurement:
    condition: str
    model: str
    device: str
    execution: Execution
    disk: DiskDelta
    residency_before: Residency
    residency_after: Residency
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["disk"] = {**asdict(self.disk), "gb_read": self.disk.gb_read,
                     "bandwidth_gbps": self.disk.bandwidth_gbps, "iops": self.disk.iops,
                     "mean_read_bytes": self.disk.mean_read_bytes,
                     "await_ms": self.disk.await_ms}
        for k in ("residency_before", "residency_after"):
            r = getattr(self, k)
            d[k] = {**asdict(r), "resident_bytes": r.resident_bytes, "fraction": r.fraction}
        return d


def measure(condition: str, model: str | os.PathLike, argv: list[str], *,
            memory_max_bytes: int | None = None, cold: bool = False,
            timeout: float = 3600.0) -> Measurement:
    """Run one condition end to end, with the cache put in a known state first."""
    model = str(model)
    device = device_for(model)
    notes: list[str] = []

    if cold:
        before = drop_cache(model)
        if before.fraction > 0.01:
            notes.append(f"cold requested but {before.fraction:.1%} stayed resident "
                         "after fadvise(DONTNEED); another process holds it")
    else:
        before = residency(model)

    d0 = disk_counters(device)
    execution = run_limited(argv, memory_max_bytes=memory_max_bytes, timeout=timeout)
    d1 = disk_counters(device)
    after = residency(model)

    if execution.timed_out:
        notes.append(f"killed at the {timeout:.0f}s budget; throughput figures are lower bounds")
    if execution.peak_memory_bytes and memory_max_bytes:
        if execution.peak_memory_bytes >= memory_max_bytes * 0.99:
            notes.append("peak reached the ceiling — the limit bound the run, as intended")
    return Measurement(condition=condition, model=model, device=device, execution=execution,
                       disk=d1 - d0, residency_before=before, residency_after=after, notes=notes)


def write_report(measurements: list[Measurement], path: str | os.PathLike) -> None:
    Path(path).write_text(json.dumps([m.to_dict() for m in measurements], indent=2))

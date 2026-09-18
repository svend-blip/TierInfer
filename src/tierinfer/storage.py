"""Reading model bytes on purpose, and measuring what that costs.

The baseline experiment this project starts from produced roughly 200 000
reads per second of about 4 KB each, because the kernel was reacting to pages
that were already missing. This module exists to do the opposite: read a named
object, in one operation, before it is needed — and to record what each read
actually cost so the claim can be checked instead of asserted.

Two facilities make the measurements honest on a machine where the page cache
holds most of the model:

* ``evict`` drops a byte range from the page cache with
  ``posix_fadvise(POSIX_FADV_DONTNEED)``. It needs no privileges, so a cold
  read can be measured without dropping every cache on the system.
* every read returns a :class:`ReadStat`, so throughput is computed from what
  happened rather than from the drive's specification.

Measured on the reference host (Samsung 990 PRO, ext4) for one 3.09 MB expert
slab: 0.46 ms warm (7.0 GB/s), 2.46 ms cold (1.3 GB/s).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .index import ByteRange

POSIX_FADV_NORMAL = 0
POSIX_FADV_RANDOM = 1
POSIX_FADV_SEQUENTIAL = 2
POSIX_FADV_WILLNEED = 3
POSIX_FADV_DONTNEED = 4

_libc = None


def _libc_handle():
    global _libc
    if _libc is None:
        name = ctypes.util.find_library("c")
        _libc = ctypes.CDLL(name, use_errno=True) if name else False
    return _libc or None


@dataclass(frozen=True)
class ReadStat:
    """What one read cost. Bandwidth is derived, never assumed."""

    nbytes: int
    seconds: float
    operations: int

    @property
    def bytes_per_second(self) -> float:
        return self.nbytes / self.seconds if self.seconds > 0 else float("inf")

    @property
    def mean_operation_bytes(self) -> float:
        return self.nbytes / self.operations if self.operations else 0.0


#: Upper edges of the operation-size histogram, in bytes. Power-of-two
#: buckets from one page to 64 MiB; the last bucket is open-ended.
SIZE_BUCKETS: tuple[int, ...] = tuple(4096 << i for i in range(15))


@dataclass
class StorageStats:
    """Running totals for one backend, for telemetry.

    Operation sizes are kept as a bounded histogram rather than a list: the
    first version appended every size to a list, which on a run of a few
    hundred thousand expert reads is a few hundred thousand integers that
    nothing ever freed.
    """

    reads: int = 0
    operations: int = 0
    bytes_read: int = 0
    seconds: float = 0.0
    size_histogram: dict[int, int] = field(default_factory=dict)

    def record(self, stat: ReadStat, sizes: list[int]) -> None:
        self.reads += 1
        self.operations += stat.operations
        self.bytes_read += stat.nbytes
        self.seconds += stat.seconds
        h = self.size_histogram
        for n in sizes:
            edge = next((b for b in SIZE_BUCKETS if n <= b), 0)   # 0 = above the last edge
            h[edge] = h.get(edge, 0) + 1

    @property
    def bytes_per_second(self) -> float:
        return self.bytes_read / self.seconds if self.seconds > 0 else 0.0

    @property
    def mean_operation_bytes(self) -> float:
        return self.bytes_read / self.operations if self.operations else 0.0


class StorageBackend:
    """Byte-range reads against one model's files, with timing and cache control.

    A model is one file or several — a ``gguf-split`` model is several — and
    a :class:`ByteRange` names the file it indexes. One descriptor is held
    per file for the backend's lifetime; a range whose ``path`` is ``None``
    means the first (or only) file, which keeps single-file callers and
    hand-built test ranges unchanged.
    """

    def __init__(self, paths: str | Path | Sequence[str | Path], *,
                 advise_random: bool = True, align: int = 0):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if align < 0 or (align & (align - 1)):
            raise ValueError("align must be zero or a power of two")
        #: Round every read outward to a multiple of this many bytes. On an md
        #: RAID0 with a 512 KiB chunk, an expert-sized read that starts 32 bytes
        #: into a chunk is split at every chunk boundary into two partial
        #: requests (md0 saw 361 KB means where the chunk is 512); reading the
        #: covering aligned range costs a little more data and lets every
        #: request be a whole chunk. Whether that pays is measured, not assumed
        #: — this is the knob the measurement turns.
        self.align = align
        files = [Path(p) for p in paths]
        if not files:
            raise ValueError("a backend needs at least one file")
        self.paths: tuple[Path, ...] = tuple(files)
        self.path = files[0]
        self._fds: dict[Path, int] = {}
        try:
            for f in files:
                self._fds[f] = os.open(f, os.O_RDONLY)
        except OSError:
            self.close()
            raise
        self.stats = StorageStats()
        if advise_random:
            # The access pattern is expert-sized jumps, not a sequential scan;
            # saying so stops the kernel reading ahead into weights nobody asked for.
            for fd in self._fds.values():
                self._fadvise(fd, 0, 0, POSIX_FADV_RANDOM)

    @classmethod
    def for_model(cls, gguf, **kw) -> "StorageBackend":
        """A backend over every file of a :class:`~tierinfer.gguf.GGUFFile`."""
        return cls(gguf.files, **kw)

    # -- lifecycle ------------------------------------------------------

    @property
    def fd(self) -> int:
        """The first file's descriptor. Kept for single-file callers."""
        return self._fds.get(self.path, -1)

    def fd_for(self, r: ByteRange) -> int:
        """The descriptor for the file this range indexes."""
        if r.path is None:
            return self.fd
        fd = self._fds.get(Path(r.path))
        if fd is None:
            raise OSError(f"{r.name} indexes {r.path}, which this backend did not open "
                          f"(it has {', '.join(p.name for p in self.paths)})")
        return fd

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "StorageBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- cache control --------------------------------------------------

    def _fadvise(self, fd: int, offset: int, length: int, advice: int) -> int:
        """``posix_fadvise``, returning its errno rather than swallowing it.

        Zero means the kernel accepted the advice. Anything else is reported
        to the caller, because an eviction that did not happen turns every
        "cold" measurement after it into a warm one that says cold.
        """
        libc = _libc_handle()
        if libc is None:
            return errno.ENOSYS
        return libc.posix_fadvise(fd, ctypes.c_long(offset), ctypes.c_long(length), advice)

    def evict(self, ranges: list[ByteRange] | ByteRange) -> None:
        """Drop these ranges from the page cache, so the next read is a real one.

        Raises ``OSError`` when the kernel refuses, rather than returning as
        if the pages were gone.
        """
        for r in _as_list(ranges):
            rc = self._fadvise(self.fd_for(r), r.file_offset, r.nbytes, POSIX_FADV_DONTNEED)
            if rc:
                raise OSError(rc, f"posix_fadvise(DONTNEED) on {r.name}: {os.strerror(rc)}")

    def hint_willneed(self, ranges: list[ByteRange] | ByteRange) -> None:
        """Ask the kernel to start fetching these. Advisory: it may do nothing.

        Measured on this project to do nothing under memory pressure
        (`benchmarks/REAL-ROUTING.md`); kept as the instrument that showed it.
        """
        for r in _as_list(ranges):
            rc = self._fadvise(self.fd_for(r), r.file_offset, r.nbytes, POSIX_FADV_WILLNEED)
            if rc:
                raise OSError(rc, f"posix_fadvise(WILLNEED) on {r.name}: {os.strerror(rc)}")

    # -- reads ----------------------------------------------------------

    def read(self, ranges: list[ByteRange] | ByteRange) -> tuple[list[bytes], ReadStat]:
        """Read these ranges, one operation each, and time the whole thing."""
        rs = _as_list(ranges)
        fds = [self.fd_for(r) for r in rs]
        if self.align:
            asks = [(r.file_offset - r.file_offset % self.align,
                     -(-(r.file_offset + r.nbytes) // self.align) * self.align) for r in rs]
        else:
            asks = [(r.file_offset, r.file_offset + r.nbytes) for r in rs]
        sizes = [b - a for a, b in asks]
        t0 = time.perf_counter()
        raw = [os.pread(fd, b - a, a) for fd, (a, b) in zip(fds, asks)]
        elapsed = time.perf_counter() - t0
        blobs = []
        for got, r, (a, _) in zip(raw, rs, asks):
            blob = got[r.file_offset - a:r.file_offset - a + r.nbytes] if self.align else got
            if len(blob) != r.nbytes:
                raise OSError(f"short read on {r.name}: {len(blob)} of {r.nbytes} bytes")
            blobs.append(blob)
        stat = ReadStat(nbytes=sum(len(g) for g in raw), seconds=elapsed, operations=len(rs))
        self.stats.record(stat, sizes)
        return blobs, stat

    def read_paged(self, ranges: list[ByteRange] | ByteRange,
                   page: int = 4096) -> tuple[int, ReadStat]:
        """Read the same bytes in page-sized pieces, the way demand paging would.

        Here to be compared against :meth:`read`, not to be used in anger. It
        answers the question the baseline experiment raised: how much of the
        cost was the bytes, and how much was asking for them 4 KB at a time.
        """
        rs = _as_list(ranges)
        sizes: list[int] = []
        total = 0
        t0 = time.perf_counter()
        for r in rs:
            fd = self.fd_for(r)
            done = 0
            while done < r.nbytes:
                n = min(page, r.nbytes - done)
                total += len(os.pread(fd, n, r.file_offset + done))
                sizes.append(n)
                done += n
        elapsed = time.perf_counter() - t0
        stat = ReadStat(nbytes=total, seconds=elapsed, operations=len(sizes))
        self.stats.record(stat, sizes)
        return total, stat


def coalesce(ranges: list[ByteRange], *, gap: int = 1 << 20) -> list[ByteRange]:
    """Merge ranges that are adjacent or separated by less than ``gap``.

    Reading across a small gap costs less than a second operation, so a plan
    that touches neighbouring slabs should ask for them once. The merged range
    is named for what it covers so telemetry stays readable.
    """
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: r.file_offset)
    out: list[ByteRange] = []
    cur_start = ordered[0].file_offset
    cur_end = ordered[0].end
    members = [ordered[0].name]
    for r in ordered[1:]:
        if r.file_offset - cur_end <= gap:
            cur_end = max(cur_end, r.end)
            members.append(r.name)
        else:
            out.append(ByteRange(_merged_name(members), cur_start, cur_end - cur_start))
            cur_start, cur_end, members = r.file_offset, r.end, [r.name]
    out.append(ByteRange(_merged_name(members), cur_start, cur_end - cur_start))
    return out


def _merged_name(members: list[str]) -> str:
    return members[0] if len(members) == 1 else f"{members[0]}+{len(members) - 1}more"


def _as_list(ranges) -> list[ByteRange]:
    return [ranges] if isinstance(ranges, ByteRange) else list(ranges)

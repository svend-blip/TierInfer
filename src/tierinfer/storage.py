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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class StorageStats:
    """Running totals for one backend, for telemetry."""

    reads: int = 0
    operations: int = 0
    bytes_read: int = 0
    seconds: float = 0.0
    operation_sizes: list[int] = field(default_factory=list)

    def record(self, stat: ReadStat, sizes: list[int]) -> None:
        self.reads += 1
        self.operations += stat.operations
        self.bytes_read += stat.nbytes
        self.seconds += stat.seconds
        self.operation_sizes.extend(sizes)

    @property
    def bytes_per_second(self) -> float:
        return self.bytes_read / self.seconds if self.seconds > 0 else 0.0

    @property
    def mean_operation_bytes(self) -> float:
        return self.bytes_read / self.operations if self.operations else 0.0


class StorageBackend:
    """Byte-range reads against one model file, with timing and cache control."""

    def __init__(self, path: str | Path, *, advise_random: bool = True):
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.stats = StorageStats()
        if advise_random:
            # The access pattern is expert-sized jumps, not a sequential scan;
            # saying so stops the kernel reading ahead into weights nobody asked for.
            self._fadvise(0, 0, POSIX_FADV_RANDOM)

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "StorageBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- cache control --------------------------------------------------

    def _fadvise(self, offset: int, length: int, advice: int) -> int:
        libc = _libc_handle()
        if libc is None:
            return -1
        return libc.posix_fadvise(
            self.fd, ctypes.c_long(offset), ctypes.c_long(length), advice
        )

    def evict(self, ranges: list[ByteRange] | ByteRange) -> None:
        """Drop these ranges from the page cache, so the next read is a real one."""
        for r in _as_list(ranges):
            self._fadvise(r.file_offset, r.nbytes, POSIX_FADV_DONTNEED)

    def hint_willneed(self, ranges: list[ByteRange] | ByteRange) -> None:
        """Ask the kernel to start fetching these. Advisory: it may do nothing."""
        for r in _as_list(ranges):
            self._fadvise(r.file_offset, r.nbytes, POSIX_FADV_WILLNEED)

    # -- reads ----------------------------------------------------------

    def read(self, ranges: list[ByteRange] | ByteRange) -> tuple[list[bytes], ReadStat]:
        """Read these ranges, one operation each, and time the whole thing."""
        rs = _as_list(ranges)
        sizes = [r.nbytes for r in rs]
        t0 = time.perf_counter()
        blobs = [os.pread(self.fd, r.nbytes, r.file_offset) for r in rs]
        elapsed = time.perf_counter() - t0
        stat = ReadStat(nbytes=sum(len(b) for b in blobs), seconds=elapsed, operations=len(rs))
        self.stats.record(stat, sizes)
        for blob, r in zip(blobs, rs):
            if len(blob) != r.nbytes:
                raise OSError(f"short read on {r.name}: {len(blob)} of {r.nbytes} bytes")
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
            done = 0
            while done < r.nbytes:
                n = min(page, r.nbytes - done)
                total += len(os.pread(self.fd, n, r.file_offset + done))
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

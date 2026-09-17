#!/usr/bin/env python3
"""Does issuing expert reads ourselves beat waiting for demand paging?

Three ways to get the same experts off the same NVMe, measured cold each
time so none of them is answered out of the page cache:

    mmap-fault   touch one byte per page of the expert through a mapping —
                 what llama.cpp's reader does when a token routes to it
    pread-sync   one pread per expert, in order, on the calling thread
    stream-N     N worker threads issuing preads into pooled buffers

The comparison only means something if every run starts from the same place,
so each one drops exactly the ranges it is about to read. Correctness is
checked, not assumed: every method must return the same bytes.

    python benchmarks/streaming.py MODEL.gguf [--layers 4] [--experts 8] \\
        [--workers 1 --workers 4 --workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.gguf import read_gguf  # noqa: E402
from tierinfer.index import ModelIndex  # noqa: E402
from tierinfer.storage import StorageBackend  # noqa: E402
from tierinfer.stream import BufferPool, ExpertStreamer, PoolExhausted  # noqa: E402

MB = 1024 * 1024


def digest(chunks) -> str:
    h = hashlib.blake2b(digest_size=16)
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def mmap_fault(path: Path, ranges) -> tuple[float, str, int]:
    """Read each range by faulting it in through a mapping, one page at a time."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        try:
            page = mmap.PAGESIZE
            total = 0
            t0 = time.perf_counter()
            parts = []
            for r in ranges:
                # Touch every page, as a reader walking the tensor would.
                for off in range(r.file_offset, r.file_offset + r.nbytes, page):
                    _ = mm[off]
                parts.append(mm[r.file_offset:r.file_offset + r.nbytes])
                total += r.nbytes
            elapsed = time.perf_counter() - t0
            return elapsed, digest(parts), total
        finally:
            mm.close()
    finally:
        os.close(fd)


def pread_sync(backend: StorageBackend, ranges) -> tuple[float, str, int]:
    t0 = time.perf_counter()
    blobs, _ = backend.read(list(ranges))
    elapsed = time.perf_counter() - t0
    return elapsed, digest(blobs), sum(len(b) for b in blobs)


def stream(backend: StorageBackend, groups, workers: int, slot: int) -> tuple[float, str, int]:
    """Submit every expert at once, then collect them in submission order."""
    pool = BufferPool(slot, min(len(groups), 2 * workers) or 1)
    with ExpertStreamer(backend, pool, workers=workers) as s:
        t0 = time.perf_counter()
        parts, total, pending = [], 0, []
        for key, rs in groups:
            while True:
                try:
                    pending.append(s.submit(key, rs))
                    break
                except PoolExhausted:                   # drain one and retry
                    done = pending.pop(0)
                    view = s.wait(done, timeout=120)
                    parts.append(bytes(view)); total += len(view)
                    s.release(done)
        for load in pending:
            view = s.wait(load, timeout=120)
            parts.append(bytes(view)); total += len(view)
            s.release(load)
        elapsed = time.perf_counter() - t0
    return elapsed, digest(parts), total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8, help="experts per layer")
    ap.add_argument("--workers", type=int, action="append", default=[])
    a = ap.parse_args()
    workers = a.workers or [1, 4, 8]

    ix = ModelIndex(read_gguf(a.model))
    print(f"{a.model.name}: {ix.expert_count} experts x {len(ix.moe_layers)} MoE layers")

    groups = []
    for layer in ix.moe_layers[:a.layers]:
        for e in range(a.experts):
            groups.append(((layer, e), ix.expert(layer, e)))
    flat = [r for _, rs in groups for r in rs]
    slot = max(sum(r.nbytes for r in rs) for _, rs in groups)
    total_mb = sum(r.nbytes for r in flat) / MB
    print(f"{len(groups)} experts, {total_mb:.0f} MB, {slot / MB:.2f} MB each, "
          f"{len(flat)} ranges\n")

    print(f"{'method':<14}{'seconds':>10}{'MB/s':>10}{'vs mmap':>10}   digest")
    results = {}

    with StorageBackend(a.model) as b:
        def cold():
            b.evict(flat)

        cold()
        secs, dg, n = mmap_fault(a.model, flat)
        results["mmap-fault"] = (secs, dg)
        base = secs
        print(f"{'mmap-fault':<14}{secs:>10.2f}{n / MB / secs:>10.0f}{1.0:>10.2f}   {dg}")

        cold()
        secs, dg, n = pread_sync(b, flat)
        results["pread-sync"] = (secs, dg)
        print(f"{'pread-sync':<14}{secs:>10.2f}{n / MB / secs:>10.0f}{base / secs:>10.2f}   {dg}")

        for w in workers:
            cold()
            secs, dg, n = stream(b, groups, w, slot)
            results[f"stream-{w}"] = (secs, dg)
            print(f"{f'stream-{w}':<14}{secs:>10.2f}{n / MB / secs:>10.0f}"
                  f"{base / secs:>10.2f}   {dg}")

    digests = {dg for _, dg in results.values()}
    print()
    if len(digests) == 1:
        print("every method returned the same bytes")
        return 0
    print("MISMATCH — the methods do not agree on the bytes:", file=sys.stderr)
    for name, (_, dg) in results.items():
        print(f"  {name:<14}{dg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The loader, end to end on a synthetic split model.

A child process is started with the shim preloaded; it maps a shard the way
llama.cpp would (read-only, offset 0) and reads bytes out of it. Every byte
it sees must be the file's, every expert it touches must have arrived as one
fault, and an eviction the server orders must take the pages away so the
next touch faults again. The shim must be built (`tools/uffd/build.sh`);
without it these tests skip, and say so.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.index import load  # noqa: E402
from tierinfer.loader import FileLayout, LoaderServer  # noqa: E402

from test_gguf import IQ4_XS, F32, write_gguf  # noqa: E402
from test_shards import _fill  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "build" / "libtierinfer_mmap.so"

CHILD = r'''
import mmap, os, sys, json, time, ctypes, ctypes.util
path, plan = sys.argv[1], json.loads(sys.argv[2])
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
out = []
for step in plan:
    if step[0] == "unmap_head":
        base = None
        for line in open(os.environ["TIERINFER_BASE_FILE"]):
            p_, b_, l_ = line.split()
            if p_ == os.path.realpath(path):
                base = int(b_, 16)
        libc = ctypes.CDLL(ctypes.util.find_library("c")); libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        assert libc.munmap(base, step[1]) == 0
    elif step[0] == "unmap_tail":
        # what llama.cpp does with fragments no used tensor lives in
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        # the shim notes where the anonymous region landed, for exactly this
        base = None
        for line in open(os.environ["TIERINFER_BASE_FILE"]):
            p_, b_, l_ = line.split()
            if p_ == os.path.realpath(path):
                base = int(b_, 16)
        assert base is not None, "the shim did not note the mapping"
        start = (size - step[1]) & ~4095
        assert libc.munmap(ctypes.c_void_p(base + start), ctypes.c_size_t(((size + 4095) & ~4095) - start)) == 0
    elif step[0] == "unmap_range":
        # what llama.cpp does with experts it has uploaded to the GPU: the
        # suffix fragment goes, and the addresses are anyone's afterwards.
        # Through the global namespace, so the preloaded interposer sees it
        # the way it sees llama.cpp's call (a libc handle would bypass it).
        libc = ctypes.CDLL(None)
        base = None
        for line in open(os.environ["TIERINFER_BASE_FILE"]):
            p_, b_, l_ = line.split()
            if p_ == os.path.realpath(path):
                base = int(b_, 16)
        assert libc.munmap(ctypes.c_void_p(base + step[1]), ctypes.c_size_t(step[2])) == 0
    elif step[0] == "read":
        _, off, n = step
        t0 = time.perf_counter(); data = mm[off:off + n]; dt = time.perf_counter() - t0
        out.append({"off": off, "n": n, "digest": __import__("hashlib").blake2b(data, digest_size=8).hexdigest(), "ms": dt * 1000})
    elif step[0] == "sleep":
        time.sleep(step[1])
print(json.dumps(out))
'''


def _sock_path(tmp_path):
    # unix socket paths are short; tmp_path can be long
    return str(Path(tempfile.mkdtemp(prefix="ti-", dir="/tmp")) / "s")


@pytest.fixture
def served(tmp_path):
    if not SHIM.exists():
        pytest.skip(f"shim not built at {SHIM} (run tools/uffd/build.sh)")
    shards = split_model(tmp_path, count=2, layers_per=2)
    ix = load(shards[0])
    yield ix, shards


EXPERTS = 4


def split_model(tmp_path, count=2, layers_per=2):
    """Like test_shards.split_model, with experts larger than a page (8 704 B):
    the kernel reports faults per page, so a slab has to be at least one."""
    total = 0
    spec = []
    for no in range(count):
        tensors = [("token_embd.weight", (256, 8), F32)] if no == 0 else []
        for l in range(no * layers_per, (no + 1) * layers_per):
            tensors += [(f"blk.{l}.attn_q.weight", (256, 8), IQ4_XS),
                        (f"blk.{l}.ffn_gate_inp.weight", (256, EXPERTS), F32),
                        (f"blk.{l}.ffn_gate_exps.weight", (256, 64, EXPERTS), IQ4_XS),
                        (f"blk.{l}.ffn_up_exps.weight", (256, 64, EXPERTS), IQ4_XS),
                        (f"blk.{l}.ffn_down_exps.weight", (256, 64, EXPERTS), IQ4_XS)]
        total += len(tensors)
        spec.append((no, tensors))
    out = []
    for no, tensors in spec:
        meta = {"split.no": no, "split.count": count, "split.tensors.count": total}
        if no == 0:
            meta.update({"general.architecture": "qwen3moe", "qwen3moe.expert_count": EXPERTS,
                         "qwen3moe.expert_used_count": 2, "qwen3moe.block_count": count * layers_per})
        p = write_gguf(tmp_path / f"m-{no + 1:05d}-of-{count:05d}.gguf", tensors, meta)
        _fill(p, seed=no + 1)
        out.append(p)
    return out


def _start(server, sock):
    t = threading.Thread(target=server.serve, args=(sock,), daemon=True)
    t.start()
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.02)
    return t


def _run_child(shard, plan, sock, files, timeout=60):
    note = str(Path(sock).parent / "bases")
    env = dict(os.environ, LD_PRELOAD=str(SHIM), TIERINFER_SOCK=sock, TIERINFER_FILES=files,
               TIERINFER_BASE_FILE=note)
    import json
    r = subprocess.run([sys.executable, "-c", CHILD, str(shard), json.dumps(plan)],
                       capture_output=True, text=True, env=env, timeout=timeout)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), r.stderr


def _digest(path, off, n):
    import hashlib
    with open(path, "rb") as fh:
        fh.seek(off)
        return hashlib.blake2b(fh.read(n), digest_size=8).hexdigest()


def test_the_layout_names_every_byte_of_a_shard(served):
    ix, shards = served
    lay = FileLayout(ix, shards[1])
    expert_regions = [r for r in lay.regions if r.key[0] != "floor"]
    assert expert_regions and all(lay.at(r.start) is r for r in expert_regions)
    assert lay.at(expert_regions[0].start - 1) is not expert_regions[0]
    ref = ix.expert(3, 2)                      # layer 3 lives in shard 2
    assert {r.key for r in lay.by_key[(3, 2)]} == {(3, 2)}
    assert sum(r.end - r.start for r in lay.by_key[(3, 2)]) == ref.nbytes


def test_bytes_through_the_shim_are_the_files_bytes_and_arrive_per_expert(served):
    ix, shards = served
    sock = _sock_path(None)
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        ref = ix.expert(1, 3)                  # in shard 1
        plan = [["read", r.file_offset, r.nbytes] for r in ref.ranges]
        floor = next(t for t in ix.gguf.tensors if t.name == "blk.0.attn_q.weight")
        plan.append(["read", floor.file_offset, floor.nbytes])
        out, err = _run_child(shards[0], plan, sock, files)
        assert "serving" in err, err
        for step, got in zip(plan, out):
            assert got["digest"] == _digest(shards[0], step[1], step[2]), (step, err)
        # three slabs of one expert: the first touch served the whole expert,
        # the other two found their pages present (or were duplicates), no second read
        assert server.stats.faults_expert >= 1
        assert (1, 3) in server.cache
        assert server.stats.faults_floor >= 1
    finally:
        server.close()


def test_an_eviction_takes_the_pages_away_and_the_next_touch_faults_again(served):
    ix, shards = served
    sock = _sock_path(None)
    one = ix.expert(0, 0).nbytes
    server = LoaderServer(ix, ram_bytes=2 * one + 4096, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        plan = []
        for e in range(EXPERTS):               # four experts through a two-expert tier
            ref = ix.expert(0, e)
            plan += [["read", r.file_offset, r.nbytes] for r in ref.ranges]
        plan.append(["sleep", 0.3])
        ref0 = ix.expert(0, 0)                 # evicted by then: must fault again, still correct
        plan += [["read", r.file_offset, r.nbytes] for r in ref0.ranges]
        out, err = _run_child(shards[0], plan, sock, files)
        reads = [s for s in plan if s[0] == "read"]
        assert len(reads) == len(out)
        for step, got in zip(reads, out):
            assert got["digest"] == _digest(shards[0], step[1], step[2]), err
        assert server.stats.evictions >= 2, (server.stats, err)
        assert server.stats.evict_bytes > 0
        assert len(server.cache) <= 2
        # expert 0 was served twice: once cold, once after its eviction
        assert server.stats.faults_expert >= 5, server.stats
    finally:
        server.close()


def test_a_chunk_that_straddles_an_unmapped_tail_is_still_served(served):
    """llama.cpp munmaps the part of a file no used tensor lives in; a floor
    chunk that reaches into it must still deliver its mapped pages, or the
    client faults on the same page forever (the first live run did)."""
    ix, shards = served
    sock = _sock_path(None)
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=16384)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        size = shards[0].stat().st_size
        # unmap the last two pages; read the whole page just before the cut
        cut = (size - 8192) & ~4095
        plan = [["unmap_tail", 8192], ["read", cut - 4096, 4096]]
        out, err = _run_child(shards[0], plan, sock, files)
        assert out[0]["digest"] == _digest(shards[0], cut - 4096, 4096), err
        assert server.stats.faults >= 1
        assert server.stats.repeat_faults == 0
    finally:
        server.close()


def test_a_chunk_whose_head_is_unmapped_still_serves_its_tail(served):
    """llama.cpp also unmaps whatever precedes the first used tensor."""
    ix, shards = served
    sock = _sock_path(None)
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=16384)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        plan = [["unmap_head", 8192], ["read", 8192, 4096]]      # first chunk: pages 0-1 gone, read page 2
        out, err = _run_child(shards[0], plan, sock, files)
        assert out[0]["digest"] == _digest(shards[0], 8192, 4096), err
        assert server.stats.repeat_faults == 0
        assert server.stats.unmapped_pages >= 1
    finally:
        server.close()


def test_a_fault_on_a_present_page_is_woken_not_recopied(served):
    """Parallel compute threads fault on pages of an expert whose copy is in
    flight; those arrive after the copy and must not cost another copy."""
    ix, shards = served
    sock = _sock_path(None)
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        ref = ix.expert(1, 3)
        r0 = ref.ranges[0]
        # read the same expert twice; the second read of a present page faults
        # nowhere, so drive a re-fault by hand through the server's own path
        plan = [["read", r0.file_offset, r0.nbytes]]
        out, err = _run_child(shards[0], plan, sock, files)
        assert out[0]["digest"] == _digest(shards[0], r0.file_offset, r0.nbytes)
        copied_before = server.stats.bytes_copied
        m = server.mappings[-1]
        m.dead = False
        # simulate the queued event: the page is gone with the client, so the
        # presence check cannot answer and the counter heuristic applies
        server._serve_expert(m, (1, 3), why="fault", offset=r0.file_offset & ~4095)
        assert server.stats.faults_resident >= 1
        assert server.stats.bytes_copied == copied_before
    finally:
        server.close()


# -- prefetch scheduling and coalescing (item 3) --------------------------------


def test_the_scheduler_serves_by_class_and_drops_stale_layers():
    import threading as _t
    from tierinfer.loader import PrefetchScheduler, Priority
    served, gate = [], _t.Event()

    def serve(keys):
        gate.wait(5)
        served.extend(keys)
    s = PrefetchScheduler(serve, workers=1, capacity=8)
    try:
        # occupy the single worker first, so the real jobs queue up and are
        # ordered by class rather than by arrival
        s.submit((0, 0), layer=0, token=0, priority=Priority.HIGH_CONFIDENCE_NEXT)
        time.sleep(0.05)
        s.submit((5, 1), layer=5, token=0, priority=Priority.PROBABLE_NEXT)
        s.submit((6, 7), layer=6, token=0, priority=Priority.HIGH_CONFIDENCE_NEXT)
        s.submit((9, 2), layer=9, token=0, priority=Priority.BACKGROUND_HOT)
        s.routed(5, 0)                     # layer 5's routing arrived: its job is stale
        gate.set()
        deadline = time.time() + 5
        while s.served + s.superseded < 4 and time.time() < deadline:
            time.sleep(0.01)
        assert served[1] == (6, 7), served             # high confidence before the rest
        assert (5, 1) not in served and s.superseded == 1
        assert (9, 2) in served
        assert s.by_class[Priority.HIGH_CONFIDENCE_NEXT] == 2
    finally:
        s.close()


def test_the_scheduler_is_bounded():
    from tierinfer.loader import PrefetchScheduler, Priority
    import threading as _t
    gate = _t.Event()
    s = PrefetchScheduler(lambda keys: gate.wait(5), workers=1, capacity=3)
    try:
        ok = [s.submit((1, i), layer=1, token=0, priority=Priority.PROBABLE_NEXT) for i in range(6)]
        assert sum(ok) <= 4 and s.dropped_full >= 2       # one may be in flight, three queued
    finally:
        gate.set()
        s.close()


def test_consecutive_experts_are_read_together(served):
    """Three neighbouring experts of one layer: one read per projection for
    the run, not one per expert — and every byte still the file's."""
    ix, shards = served
    sock = _sock_path(None)
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        # one read through the shim so a mapping exists; drive the batch path
        # while the child is still alive (a dead mapping is not served)
        r0 = ix.expert(0, 0).ranges[0]
        done = {}

        def child():
            done["r"] = _run_child(shards[0], [["read", r0.file_offset, 16], ["sleep", 1.5]], sock, files)

        t = threading.Thread(target=child)
        t.start()
        for _ in range(200):
            if server.mappings and server.stats.faults >= 1:
                break
            time.sleep(0.01)
        ops0 = server.backend.stats.operations
        server._serve_batch([(1, 1), (1, 2), (1, 3)])
        t.join(10)
        assert server.backend.stats.operations - ops0 <= len(ix.expert(1, 1).ranges), \
            "a run of three consecutive experts must not cost more reads than one expert"
        assert server.stats.coalesced_reads >= 1
    finally:
        server.close()


def test_an_unmapped_expert_is_forgotten_and_never_evicted_into(served):
    """The first 480B run: experts of the two GPU layers were served at load,
    llama.cpp then unmapped that suffix, the kernel reused the addresses for
    thread stacks, and when the tier filled the cache evicted those experts
    — MADV_DONTNEED on llama's own stacks. free(): invalid pointer.

    Now the shim reports the unmap, the server forgets what lived there and
    evicts around holes, and the shim refuses an EVICT into one regardless."""
    ix, shards = served
    sock = _sock_path(None)
    one = ix.expert(0, 0).nbytes
    server = LoaderServer(ix, ram_bytes=2 * one + 4096, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        victim = ix.expert(0, 0)
        # the interior pages of each of its three slabs (a slab is 8 704 B:
        # one whole page inside it; its edge pages belong to the neighbours)
        holes = [((r.file_offset + 4095) & ~4095, (r.file_offset + r.nbytes) & ~4095) for r in victim.ranges]
        assert all(b > a for a, b in holes), holes
        plan = [["read", r.file_offset, r.nbytes] for r in victim.ranges]     # resident
        plan += [["unmap_range", a, b - a] for a, b in holes] + [["sleep", 0.3]]   # gone
        for e in range(1, EXPERTS):                                            # fill past the tier
            plan += [["read", r.file_offset, r.nbytes] for r in ix.expert(0, e).ranges]
        plan.append(["sleep", 0.3])
        out, err = _run_child(shards[0], plan, sock, files)
        for step, got in zip([s for s in plan if s[0] == "read"], out):
            assert got["digest"] == _digest(shards[0], step[1], step[2]), err
        m = server.mappings[0]
        assert server.stats.unmaps >= 3 and m.holes[:3] == holes, (server.stats, m.holes)
        assert (0, 0) in m.gone and server.stats.forgotten == 1, (m.gone, server.stats)
        assert (0, 0) not in server.cache
        # the tier still evicted something for experts 1..3 — never expert 0
        assert server.stats.evictions >= 1, server.stats
        assert "madvise(DONTNEED" not in err, err
        assert "refusing EVICT" not in err, err
    finally:
        server.close()


def test_the_shim_refuses_an_eviction_into_a_hole(served):
    ix, shards = served
    sock = _sock_path(None)
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        files = ":".join(str(f) for f in ix.gguf.files)
        victim = ix.expert(0, 0)
        lo = min(r.file_offset for r in victim.ranges)
        hi = max(r.file_offset + r.nbytes for r in victim.ranges)
        a, b = (lo + 4095) & ~4095, hi & ~4095
        plan = [["read", r.file_offset, r.nbytes] for r in victim.ranges] + [["unmap_range", a, b - a], ["sleep", 1.5]]
        sent = {}

        def poke():
            for _ in range(200):
                if server.mappings and server.mappings[0].holes:
                    m = server.mappings[0]
                    server._evict_conns[m.pid].sendall(f"EVICT {m.base + a:x} {b - a}\n".encode())
                    sent["at"] = time.time()
                    return
                time.sleep(0.01)

        t = threading.Thread(target=poke, daemon=True)
        t.start()
        out, err = _run_child(shards[0], plan, sock, files)
        t.join(5)
        assert sent, "the server never learned of the hole"
        assert "refusing EVICT" in err, err
        assert "madvise(DONTNEED" not in err, err
    finally:
        server.close()

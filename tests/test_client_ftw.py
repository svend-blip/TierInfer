"""A Python client's buffer served from an FTW checkpoint: what FreeToken's
``HostBank(backing="tierinfer")`` will do. A child process maps a bank as a
logical slice of the model's byte region; every row it reads is the shard's
bytes; a tier too small for the layer evicts and the next read is right again.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from tierinfer.ftw import FTWIndex  # noqa: E402
from tierinfer.loader import LoaderServer  # noqa: E402
from test_ftw import ALIGN, EXPERTS, write_ftw  # noqa: E402

CHILD = r'''
import hashlib, json, sys, os
sys.path.insert(0, sys.argv[1])
from tierinfer.client import TieredRegion, session
sock, plan = sys.argv[2], json.loads(sys.argv[3])
regions, out = {}, []
for step in plan:
    if step[0] == "map":
        _, tag, nbytes, off = step
        regions[tag] = TieredRegion(sock, tag=tag, nbytes=nbytes, logical_off=off, model_dir=os.environ.get("TI_MODEL_DIR"))
    elif step[0] == "read":
        # touch through a foreign call, which drops the GIL while the page
        # faults — as FreeToken's C++ pool threads do; a Python-level copy
        # would hold the GIL and starve any Python thread serving the fault
        _, tag, a, n = step
        import ctypes
        libc = ctypes.CDLL(None)
        libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        buf = (ctypes.c_char * n)()
        libc.memcpy(buf, regions[tag].base + a, n)
        out.append(hashlib.blake2b(bytes(buf), digest_size=8).hexdigest())
    elif step[0] == "route":
        _, tag, layer, ids = step
        regions[tag].route(layer, ids)
    elif step[0] == "routed":
        _, tag, layer, ids = step
        regions[tag].route(layer, ids, after=True)
    elif step[0] == "sleep":
        import time; time.sleep(step[1])
    elif step[0] == "close":
        regions.pop(step[1]).close()
s = session(sock)
print(json.dumps({"out": out, "evictions": s.evictions, "refused": s.refused, "fallback": s.fallback, "fallback_pages": s.fallback_pages, "fallback_woken": s.fallback_woken}))
'''


def _sock():
    return str(Path(tempfile.mkdtemp(prefix="ti-", dir="/tmp")) / "s")


def _start(server, sock):
    t = threading.Thread(target=server.serve, args=(sock,), daemon=True)
    t.start()
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.02)
    return t


def _child(sock, plan, timeout=60, env=None):
    src = str(Path(__file__).resolve().parent.parent / "src")
    r = subprocess.run([sys.executable, "-c", CHILD, src, sock, json.dumps(plan)],
                       capture_output=True, text=True, timeout=timeout, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def _digest(data, a, n):
    return hashlib.blake2b(data[a:a + n], digest_size=8).hexdigest()


@pytest.fixture
def ftw(tmp_path):
    data = write_ftw(tmp_path)
    return FTWIndex(tmp_path), data


def test_a_logical_buffer_is_served_row_by_row_from_the_shards(ftw):
    ix, data = ftw
    sock = _sock()
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        gu = ix.banks[0]["gate_up_packed"]
        dn = ix.banks[0]["down_packed"]
        row_gu, row_dn = gu.nbytes // EXPERTS, dn.nbytes // EXPERTS
        plan = [["map", "gu0", gu.nbytes, gu.global_off], ["map", "dn0", dn.nbytes, dn.global_off]]
        # touch one byte of expert 2's gate_up row: the whole expert (both banks) should arrive
        plan += [["read", "gu0", 2 * row_gu + 100, 16], ["sleep", 0.1]]
        for e in range(EXPERTS):
            plan += [["read", "gu0", e * row_gu, row_gu], ["read", "dn0", e * row_dn, row_dn]]
        plan += [["sleep", 0.2]]           # the last copy lands before its bookkeeping; give ROUTE the settled books
        plan += [["route", "gu0", 0, [1, 3]], ["route", "gu0", 1, [0, 2]], ["route", "gu0", 0, [2, 2]]]
        plan += [["close", "gu0"], ["close", "dn0"]]
        res, err = _child(sock, plan)
        want = [_digest(data, gu.global_off + 2 * row_gu + 100, 16)]
        for e in range(EXPERTS):
            want += [_digest(data, gu.global_off + e * row_gu, row_gu), _digest(data, dn.global_off + e * row_dn, row_dn)]
        assert res["out"] == want, err
        s = server.stats
        assert s.faults_expert >= 1 and s.bytes_copied >= EXPERTS * (row_gu + row_dn), s
        # after the first touch the whole expert (both rows, both buffers) was present
        assert s.faults_expert < 2 * EXPERTS + 2, s
        assert s.routed == 5 and s.hits == 3 and s.misses == 2, s   # layer 0 resident, layer 1 never mapped
        assert s.tokens >= 1
        assert s.unmaps == 2 and res["refused"] == 0
        assert all(m.logical_off is not None for m in server.mappings)
    finally:
        server.close()


def test_a_tier_smaller_than_the_layer_evicts_and_rereads_correctly(ftw):
    ix, data = ftw
    sock = _sock()
    one = ix.expert_nbytes(0)
    server = LoaderServer(ix, ram_bytes=2 * one + 4096, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        gu = ix.banks[0]["gate_up_packed"]
        dn = ix.banks[0]["down_packed"]
        row_gu, row_dn = gu.nbytes // EXPERTS, dn.nbytes // EXPERTS
        plan = [["map", "gu0", gu.nbytes, gu.global_off], ["map", "dn0", dn.nbytes, dn.global_off]]
        for e in range(EXPERTS):
            plan += [["read", "gu0", e * row_gu, row_gu], ["read", "dn0", e * row_dn, row_dn]]
        plan += [["sleep", 0.3], ["read", "gu0", 0, row_gu], ["read", "dn0", 0, row_dn], ["sleep", 0.2]]
        plan += [["close", "gu0"], ["close", "dn0"]]
        res, err = _child(sock, plan)
        want = []
        for e in list(range(EXPERTS)) + [0]:
            want += [_digest(data, gu.global_off + e * row_gu, row_gu), _digest(data, dn.global_off + e * row_dn, row_dn)]
        assert res["out"] == want, err
        assert server.stats.evictions >= 2, server.stats
        assert res["evictions"] >= 2 and res["refused"] == 0, res
        assert len(server.cache) <= 2
    finally:
        server.close()


def test_routing_reported_after_the_step_scores_hits_by_faults(ftw):
    """A CUDA-graph runtime can only read the ids back after the step. ROUTED
    then means: a miss is an expert that had to be faulted in during this
    token; the same routing next token, nothing faulted, is all hits."""
    ix, data = ftw
    sock = _sock()
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        gu = ix.banks[0]["gate_up_packed"]
        row_gu = gu.nbytes // EXPERTS
        plan = [["map", "gu0", gu.nbytes, gu.global_off]]
        plan += [["read", "gu0", 0, 16], ["read", "gu0", row_gu, 16], ["sleep", 0.1]]   # experts 0 and 1 faulted in
        plan += [["routed", "gu0", 0, [0, 1]], ["routed", "gu0", 1, [0, 1]]]            # step 1, reported afterwards
        plan += [["read", "gu0", 0, 16], ["read", "gu0", row_gu, 16], ["sleep", 0.1]]   # resident: no faults
        plan += [["routed", "gu0", 0, [0, 1]], ["routed", "gu0", 1, [0, 1]], ["sleep", 0.1]]   # step 2
        plan += [["read", "gu0", 2 * row_gu, 16], ["read", "gu0", 3 * row_gu, 16], ["sleep", 0.1]]   # step 3 faults 2, 3
        plan += [["routed", "gu0", 0, [2, 3]], ["routed", "gu0", 1, [2, 3]], ["sleep", 0.1]]   # its burst starts at layer 0:
        plan += [["close", "gu0"]]                                                          # the boundary must not erase them
        res, err = _child(sock, plan)
        s = server.stats
        # step 1: layer 0's two experts were faulted this token -> misses; layer 1 never touched -> "hits" by
        # the rule (not faulted). step 2: nothing faulted -> 4 hits. step 3: experts 2, 3 faulted -> 2 misses
        # on layer 0, 2 "hits" on the untouched layer 1
        assert s.tokens >= 2, s
        assert s.misses == 4 and s.hits == 8, s
    finally:
        server.close()


def test_routing_after_the_fact_prefetches_for_the_next_step(ftw):
    """With ROUTED there is no "next layer" to run ahead of; the guess is for
    the next step. Experts routed to (but never touched) are learned, issued
    as prefetch after the burst, and the next touch finds them present."""
    ix, data = ftw
    sock = _sock()
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, depth=4, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    try:
        gu = ix.banks[0]["gate_up_packed"]
        row_gu = gu.nbytes // EXPERTS
        plan = [["map", "gu0", gu.nbytes, gu.global_off]]
        for _ in range(3):                                   # three steps routed to experts 2 and 3, untouched
            plan += [["routed", "gu0", 0, [2, 3]], ["sleep", 0.3]]
        plan += [["read", "gu0", 2 * row_gu, 16], ["read", "gu0", 3 * row_gu, 16], ["sleep", 0.1]]
        plan += [["routed", "gu0", 0, [2, 3]], ["sleep", 0.2], ["close", "gu0"]]
        res, err = _child(sock, plan)
        assert res["out"] == [_digest(data, gu.global_off + 2 * row_gu, 16), _digest(data, gu.global_off + 3 * row_gu, 16)], err
        s = server.stats
        assert s.prefetch_issued >= 2, s
        assert s.prefetch_useful >= 1, s                     # the reads found them present; the last burst scored them hits
        assert s.faults_expert == 0, s                       # nothing was ever faulted in
    finally:
        server.close()


def test_a_server_that_dies_leaves_a_client_that_serves_itself(ftw, tmp_path):
    """TI-FT-012: the runtime must continue correctly or fail explicitly. With
    the server gone, faults would stall forever; the client answers them
    from the checkpoint instead — exact bytes, no cache, and it says so."""
    ix, data = ftw
    sock = _sock()
    server = LoaderServer(ix, ram_bytes=64 * 1024 * 1024, workers=2, verbose=False,
                          drop_page_cache=False, floor_chunk=4096)
    _start(server, sock)
    gu = ix.banks[0]["gate_up_packed"]
    dn = ix.banks[0]["down_packed"]
    row_gu, row_dn = gu.nbytes // EXPERTS, dn.nbytes // EXPERTS
    plan = [["map", "gu0", gu.nbytes, gu.global_off], ["map", "dn0", dn.nbytes, dn.global_off]]
    plan += [["read", "gu0", 0, row_gu], ["sleep", 2.0]]                   # served; then the server dies
    for e in range(1, EXPERTS):
        plan += [["read", "gu0", e * row_gu, row_gu], ["read", "dn0", e * row_dn, row_dn]]
    plan += [["close", "gu0"], ["close", "dn0"]]
    import os as _os
    env = dict(_os.environ, TI_MODEL_DIR=str(ix.directory))

    def killer():
        for _ in range(300):
            if server.stats.faults_expert >= 1:
                break
            time.sleep(0.01)
        time.sleep(0.3)
        server.close()

    t = threading.Thread(target=killer, daemon=True)
    t.start()
    res, err = _child(sock, plan, env=env)
    t.join(5)
    want = [_digest(data, gu.global_off, row_gu)]
    for e in range(1, EXPERTS):
        want += [_digest(data, gu.global_off + e * row_gu, row_gu), _digest(data, dn.global_off + e * row_dn, row_dn)]
    assert res["out"] == want, err
    assert res["fallback"] is True and res["fallback_pages"] >= 1, (res, err)
    assert res["fallback_woken"] == 2, res          # both live regions woken, for faults the server took with it
    assert "went away" in err

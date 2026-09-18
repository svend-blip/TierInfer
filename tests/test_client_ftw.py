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
        regions[tag] = TieredRegion(sock, tag=tag, nbytes=nbytes, logical_off=off)
    elif step[0] == "read":
        _, tag, a, n = step
        mv = regions[tag].memoryview()
        out.append(hashlib.blake2b(bytes(mv[a:a + n]), digest_size=8).hexdigest())
    elif step[0] == "route":
        _, tag, layer, ids = step
        regions[tag].route(layer, ids)
    elif step[0] == "sleep":
        import time; time.sleep(step[1])
    elif step[0] == "close":
        regions.pop(step[1]).close()
s = session(sock)
print(json.dumps({"out": out, "evictions": s.evictions, "refused": s.refused}))
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


def _child(sock, plan, timeout=60):
    src = str(Path(__file__).resolve().parent.parent / "src")
    r = subprocess.run([sys.executable, "-c", CHILD, src, sock, json.dumps(plan)],
                       capture_output=True, text=True, timeout=timeout)
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
        assert s.routed == 5 and s.hits >= 4, s          # 2 + 2 + 1 distinct; all resident
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

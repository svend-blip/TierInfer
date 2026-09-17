#!/usr/bin/env python3
"""Keep the always-resident part of a model warm while something else runs.

This is the one residency intervention that needs no synchronisation with the
inference loop. Attention weights, the shared expert, the router and the
embeddings are touched by every token — 4.55 GB of the 56.46 GB GLM-4.5-Air —
while 51.91 GB of routed experts stream past them. Under a memory ceiling the
kernel reclaims by age, and the floor is not privileged: it ages like
anything else and gets evicted by expert traffic it will immediately have to
be re-read for.

So: touch it periodically, and let the experts take the pressure instead.

Whether that is worth anything is a question, not a claim. Run it beside
llama.cpp inside the same cgroup scope — a helper in another cgroup would
have its faults charged elsewhere and would quietly widen the ceiling it is
supposed to be working under, which would make the comparison meaningless.

    systemd-run --user --scope -p MemoryMax=32G bash -c '
        python tools/pin_floor.py MODEL.gguf --interval 2 & pin=$!
        llama-cli -m MODEL.gguf ...
        kill $pin'
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.gguf import read_gguf  # noqa: E402
from tierinfer.index import ModelIndex  # noqa: E402

GB = 1024 ** 3


def floor_ranges(ix: ModelIndex):
    """Every byte range that is not a routed expert."""
    routed = set()
    for layer in ix.moe_layers:
        for e in range(ix.expert_count):
            for r in ix.expert(layer, e).ranges:
                routed.add((r.file_offset, r.nbytes))
    out = []
    for t in ix.gguf.tensors:
        key = (t.file_offset, t.nbytes)
        if key in routed:
            continue
        # A fused expert tensor is not in `routed` as a whole, only as slabs.
        if any(t.name.endswith(s) for s in
               ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")):
            continue
        out.append((t.file_offset, t.nbytes))
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between passes over the floor")
    ap.add_argument("--chunk", type=int, default=1 << 20)
    ap.add_argument("--report", type=float, default=30.0)
    a = ap.parse_args()

    ix = ModelIndex(read_gguf(a.model))
    ranges = floor_ranges(ix)
    total = sum(n for _, n in ranges)
    print(f"pin_floor: {len(ranges)} tensors, {total / GB:.2f} GB of "
          f"{a.model.stat().st_size / GB:.2f} GB, refreshed every {a.interval:g}s",
          file=sys.stderr, flush=True)

    stop = False

    def on_signal(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    fd = os.open(a.model, os.O_RDONLY)
    passes = 0
    last_report = time.monotonic()
    try:
        while not stop:
            t0 = time.monotonic()
            for off, n in ranges:
                if stop:
                    break
                read = 0
                while read < n:
                    got = os.pread(fd, min(a.chunk, n - read), off + read)
                    if not got:
                        break
                    read += len(got)
            passes += 1
            now = time.monotonic()
            if now - last_report >= a.report:
                print(f"pin_floor: {passes} passes, last took {now - t0:.1f}s",
                      file=sys.stderr, flush=True)
                last_report = now
            sleep = a.interval - (now - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        os.close(fd)
    print(f"pin_floor: stopped after {passes} passes", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

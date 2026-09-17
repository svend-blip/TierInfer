#!/usr/bin/env python3
"""What the card can hold, and what the misses cost.

The budget says how many experts fit alongside the KV cache for a context the
caller actually wants. The routing trace says which experts a real generation
asks for. Putting the two together answers goal 9's question directly: run a
56 GB model with 32 GB of VRAM, and how often does a token have to wait for a
transfer?

    python benchmarks/vram.py MODEL.gguf TRACE.jsonl [--context 16384]

Transfers are real: pinned host memory, `cudaMemcpy`, timed. The bytes are
not the model's own — replaying a trace does not need the right weights, only
the right sizes and the right sequence — but the cost of moving them is.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from tierinfer.gguf import read_gguf  # noqa: E402
from tierinfer.index import ModelIndex  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402
from tierinfer.trace import describe, read_trace  # noqa: E402
from tierinfer.vram import (  # noqa: E402
    GB, MB, CudaError, CudaRuntime, CudaUnavailable, VramBudget, VramResidency,
)

# Measured in benchmarks/BASELINE.md: the warm, unconstrained token time.
WARM_TOKEN_SECONDS = 1.0 / 5.90


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--context", type=int, action="append", default=[])
    ap.add_argument("--max-slots", type=int, default=0,
                    help="cap the pool, for a host with other work on the card")
    a = ap.parse_args()
    contexts = a.context or [4096, 16384, 32768]

    try:
        rt = CudaRuntime()
    except CudaUnavailable as e:
        print(f"no CUDA on this host: {e}", file=sys.stderr)
        return 2

    ix = ModelIndex(read_gguf(a.model))
    expert_bytes = ix.expert_nbytes()
    floor = ix.always_resident_nbytes()
    info = describe(a.trace)
    rows = read_trace(a.trace, prompt=False)
    tokens = [[(l, e) for l, es in r.routing.items() for e in es] for r in rows]

    mem = rt.memory_info()
    print(f"{a.model.name}: floor {floor / GB:.2f} GB, experts {expert_bytes / MB:.2f} MB each")
    print(f"card: {mem.total / GB:.2f} GB total, {mem.used / GB:.2f} GB already in use")
    print(f"{a.trace.name}: {len(tokens)} generated tokens, "
          f"{sum(len(t) for t in tokens) // len(tokens)} expert activations per token\n")

    print(f"{'context':>9}{'KV':>8}{'weights':>9}{'slots':>8}{'resident':>10}"
          f"{'hit rate':>10}{'moved/token':>13}{'ms/token':>10}{'of warm':>9}")

    for ctx in contexts:
        budget = VramBudget.from_model(ix.gguf.metadata, total_bytes=mem.total,
                                       context_length=ctx, layers=ix.block_count,
                                       reserve_bytes=mem.used)
        slots = budget.experts(expert_bytes, floor)
        if a.max_slots:
            slots = min(slots, a.max_slots)
        if slots <= 0:
            print(f"{ctx:>9}{budget.kv_cache / GB:>8.2f}{budget.weights / GB:>9.2f}"
                  f"{0:>8}{'—':>10}{'—':>10}{'—':>13}{'—':>10}{'—':>9}")
            continue

        tracker = ExpertTracker(window=128)
        host = rt.host_alloc(expert_bytes)
        ctypes.memset(ctypes.c_void_p(host), 0x5A, expert_bytes)
        try:
            with VramResidency(rt, expert_bytes, slots, tracker) as res:
                for token in tokens:
                    tracker.begin_token()
                    for key in token:
                        if res.lookup(key) is None:
                            res.admit(key, host, expert_bytes)
                    tracker.record(token)
                s = res.stats
                per_token = s.transfers / len(tokens)
                ms = s.transfer_seconds / len(tokens) * 1000
                print(f"{ctx:>9}{budget.kv_cache / GB:>8.2f}{budget.weights / GB:>9.2f}"
                      f"{slots:>8}{slots * expert_bytes / GB:>9.1f}G{s.hit_rate:>10.1%}"
                      f"{per_token:>13.0f}{ms:>10.1f}"
                      f"{ms / 1000 / WARM_TOKEN_SECONDS:>8.0%}")
        finally:
            rt.host_free(host)

    print(f"\nwarm token time is {WARM_TOKEN_SECONDS * 1000:.0f} ms (5.90 t/s, "
          "measured unconstrained in BASELINE.md); the last column is what the "
          "transfers would add to it")
    print("transfers are pinned host to device and really happen; the bytes are "
          "not the model's own, only its sizes and its sequence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

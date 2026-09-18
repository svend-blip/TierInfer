#!/usr/bin/env python3
"""Replay real routing through the real mechanisms against the real files.

This is the closest thing the project has to a TierInfer run under a model,
and the audit (`docs/AUDIT-2026-09-18.md`) says exactly how close: the
routing is real (captured out of the running model by `tools/trace`), the
files are real (the shards on the device under test), the reads are real
(`tierinfer.stream` into pooled buffers), the RAM cache holds the real bytes,
and the VRAM tier does real `cudaMemcpy` into real allocations. What is
absent is the compute — no kernel consumes the bytes — so the token loop has
nothing to overlap I/O with unless it is told how long a layer takes, and the
`--attn-ms` / `--ffn-ms` arms are exactly that: sleeps of a measured length,
declared as such in the output.

Every number this writes is observed, not modelled. The md0 and member-device
counters come from /proc/diskstats around and during the run, so TierInfer's
logical requests, md's merged requests and the drives' physical requests are
all on the same table (addendum §15).

    python benchmarks/replay.py SHARD.gguf TRACE.jsonl --label cache --depth 0 --ram-gb 100
    python benchmarks/replay.py SHARD.gguf TRACE.jsonl --label pf16 --depth 16 --attn-ms 4 --ffn-ms 4

The RAM tier question. `pread` goes through the page cache, so without
`--drop-after-read` the kernel keeps a second copy of everything TierInfer
reads and a TierInfer miss may be served from RAM anyway — which the md0
counters would reveal, but which makes the cache's own hit rate meaningless
as a statement about the RAM tier. With it, every delivered range is
evicted from the page cache after use, so TierInfer's cache is the only RAM
tier and every miss it reports is a read the device actually served. That
is the arm to compare against native.

Failure injection (addendum §18) is a flag, not a separate script, so it
runs through the identical path: a predictor that is always wrong, a backend
whose speculative reads fail one time in fifty, a pool of two slots, a
resident tier of eight, an expert whose ranges cannot be resolved. In every
arm a sample of delivered experts is compared byte for byte against an
exact read, and a mismatch ends the run — that is the one outcome the
project cannot have.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from tierinfer.autoconfig import Host, configure  # noqa: E402
from tierinfer.bench import (device_for, disk_counters, drop_cache, member_devices,  # noqa: E402
                             residency_of)
from tierinfer.cache import ExpertCache  # noqa: E402
from tierinfer.gguf import GGUFError  # noqa: E402
from tierinfer.index import load  # noqa: E402
from tierinfer.predict import (AdaptiveBlend, Frequency, Persistence, Prediction,  # noqa: E402
                               Predictor, Transition)
from tierinfer.prefetch import Prefetcher  # noqa: E402
from tierinfer.safety import exact_load  # noqa: E402
from tierinfer.storage import StorageBackend  # noqa: E402
from tierinfer.stream import BufferPool, ExpertStreamer  # noqa: E402
from tierinfer.telemetry import Telemetry  # noqa: E402
from tierinfer.trace import read_trace  # noqa: E402
from tierinfer.tracker import ExpertTracker  # noqa: E402

GB = 1024 ** 3
MB = 1024 ** 2


# -- failure injection --------------------------------------------------


class AlwaysWrong(Predictor):
    """Predicts the highest-numbered experts, whatever the model does."""

    name = "always-wrong"

    def __init__(self, n_expert: int) -> None:
        self.n = n_expert

    def observe(self, routing):
        pass

    def score(self, layer, sofar):
        return {e: float(self.n - e) for e in range(self.n - 16, self.n)}


class FaultyBackend(StorageBackend):
    """Fails one speculative read in ``every``. The exact path is untouched.

    Speculative reads run on the streamer's worker threads; the exact path
    runs on the caller's. Telling them apart by thread is what keeps the
    injection where the addendum wants it — on the path that is allowed to
    fail — and off the one that is not.
    """

    def __init__(self, *a, every: int = 50, **kw) -> None:
        super().__init__(*a, **kw)
        self.every = every
        self._n = 0
        self.injected = 0

    def fd_for(self, r):
        if threading.current_thread().name.startswith("tierinfer-stream"):
            self._n += 1
            if self._n % self.every == 0:
                self.injected += 1
                raise OSError(5, f"injected read failure on {r.name}")
        return super().fd_for(r)


# -- the run ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="any shard of the model")
    ap.add_argument("trace", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/replay-out"))
    ap.add_argument("--tokens", type=int, default=0, help="generated tokens to replay (0 = all)")
    ap.add_argument("--predictor-warmup", type=int, default=0,
                    help="show the predictor this many leading tokens of the trace (routing only, "
                         "no I/O) before replaying the next --tokens; with fewer than ~20 tokens of "
                         "history every prediction names an expert the last token just loaded, so "
                         "nothing speculative is ever issued and no injection on that path can fire")
    ap.add_argument("--ram-gb", type=float, default=0.0,
                    help="RAM expert cache in GiB (0 = autoconfig's share)")
    ap.add_argument("--depth", type=int, default=0, help="prefetch depth per layer (0 = cache mode)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pool-slots", type=int, default=0, help="stream buffers (0 = 4*depth+16)")
    ap.add_argument("--vram", action="store_true", help="also run the VRAM tier with real transfers")
    ap.add_argument("--vram-slots", type=int, default=0, help="override the budgeted slot count")
    ap.add_argument("--context", type=int, default=4096, help="context the VRAM budget is derived for")
    ap.add_argument("--attn-ms", type=float, default=0.0,
                    help="emulated attention time per layer, slept after before_layer")
    ap.add_argument("--ffn-ms", type=float, default=0.0,
                    help="emulated expert-FFN time per layer, slept after on_routing")
    ap.add_argument("--drop-after-read", action="store_true",
                    help="evict delivered ranges from the page cache: TierInfer's cache is the RAM tier")
    ap.add_argument("--cold", action="store_true", help="drop the model's page cache first")
    ap.add_argument("--verify-every", type=int, default=0,
                    help="compare every Nth delivered expert with an exact read (0 = off)")
    ap.add_argument("--inject", choices=["none", "bad-predictor", "fail-reads", "tiny-pool",
                                         "tiny-vram", "missing-range"], default="none")
    ap.add_argument("--snapshot-every", type=int, default=10, help="tokens between telemetry snapshots")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.out / a.label

    # -- model, host, configuration ------------------------------------
    ix = load(a.model)
    n_expert = ix.expert_count
    expert_bytes = ix.expert_nbytes_max()      # slots hold the largest; layers differ (Q6_K vs Q4_K down)
    floor = ix.always_resident_nbytes()
    host = Host.measure()
    cfg = configure(ix, context_length=a.context, host=host, stream_workers=a.workers,
                    prefetch_depth=a.depth)
    ram_bytes = int(a.ram_gb * GB) if a.ram_gb > 0 else cfg.ram_bytes
    ram_slots = ram_bytes // expert_bytes

    rows_all = read_trace(a.trace, prompt=False)
    warm_rows = rows_all[:a.predictor_warmup]
    rows = rows_all[a.predictor_warmup:]
    if a.tokens > 0:
        rows = rows[:a.tokens]
    if not rows:
        print("the trace has no generated tokens", file=sys.stderr)
        return 2

    device = device_for(ix.gguf.files[0])
    members = member_devices(device)

    if a.cold:
        r = drop_cache(a.model)
        print(f"cold: residency after drop {r.fraction:.2%}")
    res0 = residency_of(ix.gguf.files)

    # -- the mechanisms, real ------------------------------------------
    Backend = FaultyBackend if a.inject == "fail-reads" else StorageBackend
    backend = Backend.for_model(ix.gguf)
    pool_slots = a.pool_slots or (4 * a.depth + 16)
    if a.inject == "tiny-pool":
        pool_slots = 2
    pool = BufferPool(expert_bytes, pool_slots)
    streamer = ExpertStreamer(backend, pool, workers=a.workers)
    tracker = ExpertTracker(window=128)
    cache = ExpertCache(ram_bytes, tracker)
    if a.inject == "bad-predictor":
        predictor: Predictor = AlwaysWrong(n_expert)
    else:
        predictor = AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16)

    # The unresolvable expert has to be one the replay will actually route to,
    # or the injection never fires (the first attempt picked (3, 7) and the
    # eight tokens replayed never asked for it): take the first expert the
    # second replayed token routes to in its lowest MoE layer.
    broken = None
    if a.inject == "missing-range":
        r1 = rows[min(1, len(rows) - 1)]
        layer = min(r1.routing)
        broken = (layer, r1.routing[layer][0])
        print(f"missing-range: expert {broken} will not resolve; token 1 routes to it")

    def ranges_for(key):
        if key == broken:
            raise GGUFError(f"injected: expert {key} cannot be resolved to byte ranges")
        return list(ix.expert(*key).ranges)

    pf = Prefetcher(streamer, cache, predictor, ranges_for, tracker=tracker, depth=a.depth)
    for r in warm_rows:
        predictor.observe(r.as_mapping())

    vram = None
    staging = 0
    rt = None
    if a.vram or a.inject == "tiny-vram":
        from tierinfer.vram import CudaRuntime, VramBudget, VramResidency
        rt = CudaRuntime()
        mem = rt.memory_info()
        budget = VramBudget.from_model(ix.gguf.metadata, total_bytes=mem.total,
                                       context_length=a.context, layers=ix.block_count,
                                       reserve_bytes=mem.used)
        slots = budget.experts(expert_bytes, floor)
        if a.vram_slots:
            slots = a.vram_slots
        if a.inject == "tiny-vram":
            slots = 8
        if slots <= 0:
            print(f"VRAM budget leaves no slot for an expert:\n{budget.explain()}", file=sys.stderr)
            return 2
        # VramResidency defines __len__, so `if vram:` is false while it is
        # empty — including after close(). Every test below is `is not None`;
        # the first tiny-vram arm lost its VRAM numbers to exactly that.
        vram = VramResidency(rt, expert_bytes, slots, tracker)
        staging = rt.host_alloc(expert_bytes)
        print(f"vram: {slots} slots of {expert_bytes / MB:.1f} MB = {vram.device_bytes / GB:.1f} GB "
              f"(budget left {budget.weights / GB:.1f} GB after KV {budget.kv_cache / GB:.2f} GB "
              f"at {a.context}, reserve {budget.reserve / GB:.2f} GB measured)")

    tel = Telemetry(stem.with_suffix(".telemetry.jsonl"), run=a.label)
    tel.open_run(model=str(a.model), trace=str(a.trace), tokens=len(rows), depth=a.depth,
                 predictor_warmup=len(warm_rows), first_token_index=len(warm_rows),
                 ram_bytes=ram_bytes, ram_slots=ram_slots, expert_bytes=expert_bytes,
                 floor_bytes=floor, workers=a.workers, pool_slots=pool_slots,
                 vram_slots=(vram.slots if vram is not None else 0), attn_ms=a.attn_ms, ffn_ms=a.ffn_ms,
                 drop_after_read=a.drop_after_read, cold=a.cold, inject=a.inject,
                 device=device, members=",".join(members),
                 residency_before=round(res0.fraction, 4))

    print(f"replay {a.label}: {len(rows)} tokens, {len(ix.moe_layers)} MoE layers, "
          f"{ix.expert_used_count}x{len(ix.moe_layers)} = {ix.expert_used_count * len(ix.moe_layers)} "
          f"experts/token of {expert_bytes / MB:.1f} MB; RAM cache {ram_bytes / GB:.1f} GB = "
          f"{ram_slots} experts ({ram_slots / (n_expert * len(ix.moe_layers)):.1%} of all); "
          f"depth {a.depth}, {a.workers} workers, pool {pool_slots}; "
          f"{'drop-after-read' if a.drop_after_read else 'page cache free to help'}; "
          f"device {device} members {members}")

    # -- the loop ------------------------------------------------------
    d_before = disk_counters(device)
    m_before = {m: disk_counters(m) for m in members}
    per_token: list[dict] = []
    verified = mismatches = 0
    deliveries = 0
    sources = {"storage": backend.stats, "stream": streamer.stats, "cache": cache.stats,
               "prefetch": pf.stats}
    if vram is not None:
        sources["vram"] = vram.stats
    fatal = None
    t_run0 = time.perf_counter()
    try:
        for i, row in enumerate(rows):
            t0 = time.perf_counter()
            d0 = disk_counters(device)
            s0 = (pf.stats.stalls, pf.stats.late, pf.stats.used, pf.stats.issued,
                  pf.stats.wasted_bytes, cache.stats.hits, cache.stats.misses,
                  cache.stats.evictions, streamer.stats.bytes_read,
                  vram.stats.hits if vram is not None else 0, vram.stats.misses if vram is not None else 0,
                  vram.stats.bytes_transferred if vram is not None else 0,
                  vram.stats.transfer_seconds if vram is not None else 0.0)
            wait_s = vram_s = 0.0
            sofar: dict = {}
            for layer in sorted(row.routing):
                pf.before_layer(layer, sofar)
                if a.attn_ms:
                    time.sleep(a.attn_ms / 1000.0)
                tw = time.perf_counter()
                delivered = pf.on_routing(layer, row.routing[layer])
                wait_s += time.perf_counter() - tw
                for key, data in delivered.items():
                    deliveries += 1
                    if a.verify_every and deliveries % a.verify_every == 0:
                        want = exact_load(backend, ranges_for(key))
                        verified += 1
                        if data != want:
                            mismatches += 1
                            fatal = f"expert {key} delivered {len(data)} bytes that differ from the file"
                            raise RuntimeError(fatal)
                    if vram is not None:
                        tv = time.perf_counter()
                        if vram.lookup(key) is None:
                            ctypes.memmove(staging, data, len(data))
                            vram.admit(key, staging, len(data))
                        vram_s += time.perf_counter() - tv
                if a.drop_after_read:
                    for key in delivered:
                        backend.evict(ranges_for(key))
                if a.ffn_ms:
                    time.sleep(a.ffn_ms / 1000.0)
                sofar[layer] = row.routing[layer]
            pf.end_token()
            predictor.observe(row.as_mapping())
            d1 = disk_counters(device)
            s1 = (pf.stats.stalls, pf.stats.late, pf.stats.used, pf.stats.issued,
                  pf.stats.wasted_bytes, cache.stats.hits, cache.stats.misses,
                  cache.stats.evictions, streamer.stats.bytes_read,
                  vram.stats.hits if vram is not None else 0, vram.stats.misses if vram is not None else 0,
                  vram.stats.bytes_transferred if vram is not None else 0,
                  vram.stats.transfer_seconds if vram is not None else 0.0)
            dd = d1 - d0
            rec = {"token": i, "wall_ms": (time.perf_counter() - t0) * 1000,
                   "wait_ms": wait_s * 1000, "vram_ms": vram_s * 1000,
                   "stalls": s1[0] - s0[0], "late": s1[1] - s0[1],
                   "used": s1[2] - s0[2], "issued": s1[3] - s0[3],
                   "wasted_bytes": s1[4] - s0[4],
                   "cache_hits": s1[5] - s0[5], "cache_misses": s1[6] - s0[6],
                   "evictions": s1[7] - s0[7], "stream_bytes": s1[8] - s0[8],
                   "vram_hits": s1[9] - s0[9], "vram_misses": s1[10] - s0[10],
                   "vram_bytes": s1[11] - s0[11], "vram_transfer_ms": (s1[12] - s0[12]) * 1000,
                   "dev_reads": dd.reads, "dev_bytes": dd.bytes_read,
                   "dev_mean_read_bytes": dd.mean_read_bytes, "dev_await_ms": dd.await_ms,
                   "orphans": pf.orphans, "pool_in_use": pool.in_use}
            per_token.append(rec)
            tel.event("token", **rec)
            if (i + 1) % a.snapshot_every == 0:
                tel.snapshot(sources)
            if i < 3 or (i + 1) % 25 == 0:
                print(f"  token {i:>4}  {rec['wall_ms']:7.0f} ms  wait {rec['wait_ms']:6.0f}  "
                      f"stalls {rec['stalls']:3d} late {rec['late']:3d} useful {rec['used'] - rec['late']:3d}  "
                      f"cache {rec['cache_hits']:3d}/{rec['cache_hits'] + rec['cache_misses']:3d}  "
                      f"dev {rec['dev_bytes'] / GB:5.2f} GB {rec['dev_reads']:6d} reads "
                      f"@{rec['dev_mean_read_bytes'] / 1024:5.0f} KB"
                      + (f"  vram {rec['vram_hits']}/{rec['vram_hits'] + rec['vram_misses']}" if vram is not None else ""))
    except RuntimeError as e:
        fatal = str(e)
        print(f"FATAL: {fatal}", file=sys.stderr)
    except GGUFError as e:
        fatal = f"explicit failure: {e}"
        print(f"FATAL: {fatal}", file=sys.stderr)
    finally:
        run_s = time.perf_counter() - t_run0
        tel.snapshot(sources)
        d_after = disk_counters(device)
        m_after = {m: disk_counters(m) for m in members}
        res1 = residency_of(ix.gguf.files)
        # tear down in the order the buffers flow
        pf.drop_unused()
        streamer.close()
        if vram is not None:
            vram.close()
            rt.host_free(staging)
        backend.close()

    # -- the summary ---------------------------------------------------
    dev = d_after - d_before
    mem_delta = {m: (m_after[m] - m_before[m]) for m in members}
    n = len(per_token)

    def med(key):
        vals = [r[key] for r in per_token]
        return statistics.median(vals) if vals else 0.0

    def tot(key):
        return sum(r[key] for r in per_token)

    summary = {
        "label": a.label, "model": str(a.model), "trace": str(a.trace),
        "tokens_replayed": n, "predictor_warmup": len(warm_rows), "run_seconds": run_s, "fatal": fatal,
        "config": {"ram_bytes": ram_bytes, "ram_slots": ram_slots, "depth": a.depth,
                   "workers": a.workers, "pool_slots": pool_slots, "attn_ms": a.attn_ms,
                   "ffn_ms": a.ffn_ms, "drop_after_read": a.drop_after_read, "cold": a.cold,
                   "inject": a.inject, "vram_slots": vram.slots if vram is not None else 0,
                   "expert_bytes": expert_bytes, "experts_per_token":
                   ix.expert_used_count * len(ix.moe_layers)},
        "per_token": {"wall_ms_median": med("wall_ms"),
                      "wall_ms_min": min((r["wall_ms"] for r in per_token), default=0),
                      "wall_ms_max": max((r["wall_ms"] for r in per_token), default=0),
                      "wait_ms_median": med("wait_ms"),
                      "dev_bytes_median": med("dev_bytes"), "dev_reads_median": med("dev_reads"),
                      "stream_bytes_median": med("stream_bytes")},
        "prefetch": {"issued": pf.stats.issued, "used": pf.stats.used, "useful": pf.stats.useful,
                     "late": pf.stats.late, "late_wait_seconds": pf.stats.late_wait_seconds,
                     "cancelled": pf.stats.cancelled, "stalls": pf.stats.stalls,
                     "stalls_avoided": pf.stats.stalls_avoided,
                     "accuracy": pf.stats.accuracy, "mean_lead_seconds": pf.stats.mean_lead_seconds,
                     "wasted_bytes": pf.stats.wasted_bytes, "pool_exhausted": pf.stats.pool_exhausted,
                     "exact_fallbacks": pf.stats.exact_fallbacks, "timed_out": pf.stats.timed_out,
                     "orphans_at_end": pf.orphans},
        "cache": {"hits": cache.stats.hits, "misses": cache.stats.misses,
                  "hit_rate": cache.stats.hit_rate, "insertions": cache.stats.insertions,
                  "evictions": cache.stats.evictions, "bytes_admitted": cache.stats.bytes_admitted,
                  "bytes_evicted": cache.stats.bytes_evicted, "resident_end": len(cache),
                  "used_bytes_end": cache.used_bytes, "heap_fallbacks": cache.heap_fallbacks},
        "stream": {"submitted": streamer.stats.submitted, "completed": streamer.stats.completed,
                   "failed": streamer.stats.failed, "cancelled": streamer.stats.cancelled,
                   "bytes_read": streamer.stats.bytes_read,
                   "bytes_per_second": streamer.stats.bytes_per_second,
                   "mean_read_ms": streamer.stats.mean_read_seconds * 1000,
                   "mean_queue_ms": streamer.stats.mean_queue_seconds * 1000},
        "storage": {"reads": backend.stats.reads, "operations": backend.stats.operations,
                    "bytes_read": backend.stats.bytes_read,
                    "size_histogram": {str(k): v for k, v in sorted(backend.stats.size_histogram.items())},
                    "injected_failures": getattr(backend, "injected", 0)},
        "vram": ({"slots": vram.slots, "hits": vram.stats.hits, "misses": vram.stats.misses,
                  "hit_rate": vram.stats.hit_rate, "transfers": vram.stats.transfers,
                  "evictions": vram.stats.evictions,
                  "bytes_transferred": vram.stats.bytes_transferred,
                  "gbps": vram.stats.bytes_per_second / GB,
                  "mean_transfer_ms": vram.stats.mean_transfer_seconds * 1000}
                 if vram is not None else None),
        "device": {"name": device, "reads": dev.reads, "bytes": dev.bytes_read,
                   "mean_read_bytes": dev.mean_read_bytes, "await_ms": dev.await_ms,
                   "bandwidth_gbps": dev.bandwidth_gbps, "iops": dev.iops,
                   "members": {m: {"reads": x.reads, "bytes": x.bytes_read,
                                   "mean_read_bytes": x.mean_read_bytes, "await_ms": x.await_ms}
                               for m, x in mem_delta.items()}},
        "residency": {"before": res0.fraction, "after": res1.fraction},
        "verification": {"verified": verified, "mismatches": mismatches},
        "predictor": predictor.name,
        "adaptive_weights": (predictor.weights() if isinstance(predictor, AdaptiveBlend) else None),
    }
    stem.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    tel.close_run(tokens=n, fatal=fatal or "")
    tel.close()

    print(f"\n{a.label}: {n} tokens in {run_s:.0f} s, median {summary['per_token']['wall_ms_median']:.0f} ms/token"
          f" (min {summary['per_token']['wall_ms_min']:.0f}, max {summary['per_token']['wall_ms_max']:.0f})")
    print(f"  cache hit rate {cache.stats.hit_rate:.1%} ({cache.stats.hits}/{cache.stats.lookups}), "
          f"{cache.stats.evictions} evictions, {len(cache)} resident at end")
    print(f"  prefetch issued {pf.stats.issued}, useful {pf.stats.useful}, late {pf.stats.late}, "
          f"wasted {pf.stats.cancelled} ({pf.stats.wasted_bytes / GB:.1f} GB), stalls {pf.stats.stalls}, "
          f"accuracy {pf.stats.accuracy:.1%}, pool exhausted {pf.stats.pool_exhausted}, "
          f"timed out {pf.stats.timed_out}")
    print(f"  stream {streamer.stats.bytes_read / GB:.1f} GB at {streamer.stats.bytes_per_second / GB:.2f} GB/s, "
          f"{streamer.stats.failed} failed reads, storage ops {backend.stats.operations}")
    print(f"  {device}: {dev.bytes_read / GB:.1f} GB in {dev.reads} reads @ {dev.mean_read_bytes / 1024:.0f} KB, "
          f"await {dev.await_ms:.2f} ms, {dev.bandwidth_gbps:.2f} GB/s; "
          + "; ".join(f"{m}: {x.bytes_read / GB:.1f} GB in {x.reads} @ {x.mean_read_bytes / 1024:.0f} KB"
                      for m, x in mem_delta.items()))
    print(f"  per token: median {summary['per_token']['dev_bytes_median'] / GB:.2f} GB from {device} "
          f"in {summary['per_token']['dev_reads_median']:.0f} reads")
    if vram is not None:
        print(f"  vram: {vram.stats.hit_rate:.1%} hit ({vram.stats.hits}/{vram.stats.lookups}), "
              f"{vram.stats.transfers} transfers, {vram.stats.bytes_transferred / GB:.1f} GB at "
              f"{vram.stats.bytes_per_second / GB:.1f} GB/s")
    print(f"  verified {verified} deliveries against exact reads, {mismatches} mismatches; "
          f"residency {res0.fraction:.1%} -> {res1.fraction:.1%}"
          + (f"; injected failures {backend.injected}" if hasattr(backend, "injected") else ""))
    if fatal:
        print(f"  ENDED EARLY: {fatal}")
    return 1 if (fatal and a.inject != "missing-range") else 0


if __name__ == "__main__":
    raise SystemExit(main())

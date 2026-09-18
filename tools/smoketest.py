#!/usr/bin/env python3
"""Every SCOPE goal, exercised once, on the real host and the real model.

The unit tests prove the code does what it says without a 56 GB file or a
card. This does the opposite: it runs each goal's claim once against the
actual model, the actual NVMe and the actual GPU, and reports a line per
goal. It is the check that the pieces still fit together after all of them
have been changed separately.

    python tools/smoketest.py [--model MODEL.gguf] [--trace TRACE.jsonl]

Each goal reports PASS, SKIP or FAIL. A SKIP is a fact about the host — no
card, no model file — and is not a failure; a FAIL is. The exit code is
non-zero only if something failed.

Nothing here is a benchmark. The numbers it prints are there so that a change
that quietly halves a rate is visible, not so they can be quoted.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

GB = 1024 ** 3
MB = 1024 ** 2

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"
results: list[tuple[int, str, str, str]] = []


def goal(number: int, name: str):
    """Run one goal's check, catching anything it throws."""
    def wrap(fn):
        started = time.perf_counter()
        try:
            status, detail = fn()
        except _Skip as e:
            status, detail = SKIP, str(e)
        except Exception as e:                  # noqa: BLE001 — reported, not raised
            status = FAIL
            detail = f"{type(e).__name__}: {e}"
            if "--traceback" in sys.argv:
                traceback.print_exc()
        elapsed = time.perf_counter() - started
        results.append((number, name, status, detail))
        print(f"  {status}  {number:>2}. {name:<34} {detail}  [{elapsed:.1f}s]")
        return fn
    return wrap


class _Skip(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path,
                    default=Path.home() / "models" / "GLM-4.5-Air-Derestricted-IQ4_XS"
                    / "GLM-4.5-Air-Derestricted.IQ4_XS.gguf")
    ap.add_argument("--trace", type=Path, default=ROOT / "traces" / "glm45air-400.jsonl")
    ap.add_argument("--traceback", action="store_true")
    a = ap.parse_args()

    print(f"TierInfer smoke test — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"model {a.model}")
    print(f"trace {a.trace}\n")

    from tierinfer.index import load

    index = None
    if a.model.exists():
        index = load(a.model)

    def need_model():
        if index is None:
            raise _Skip(f"no model at {a.model}")
        return index

    def need_trace():
        if not a.trace.exists():
            raise _Skip(f"no trace at {a.trace}")
        return a.trace

    # -- 1, 2: layout and expert addressing -----------------------------

    @goal(2, "GGUF layout and expert ranges")
    def _():
        ix = need_model()
        for shard in ix.gguf.files:
            mine = [t for t in ix.gguf.tensors if t.path == shard]
            last = max(mine, key=lambda t: t.file_offset + t.nbytes)
            end = last.file_offset + last.nbytes
            size = shard.stat().st_size
            if end != size:
                return FAIL, f"{shard.name}: last tensor ends at {end}, file is {size}"
        ref = ix.expert(ix.moe_layers[0], 0)
        return PASS, (f"{len(ix.gguf.tensors)} tensors in {len(ix.gguf.files)} file(s), "
                      f"each ends exactly at EOF, expert = {len(ref.ranges)} ranges of "
                      f"{ref.nbytes / MB:.2f} MB")

    # -- 3: baseline measurement primitives -----------------------------

    @goal(3, "baseline primitives, no root")
    def _():
        from tierinfer.bench import device_for, disk_counters, drop_cache, residency
        ix = need_model()
        dev = device_for(a.model)
        before = disk_counters(dev)
        cold = drop_cache(a.model)
        with open(a.model, "rb") as fh:
            fh.seek(1 * GB)
            fh.read(64 * MB)
        after = disk_counters(dev)
        warm = residency(a.model)
        delta = after - before
        if delta.bytes_read <= 0:
            return FAIL, "the device counters did not move for a 64 MB read"
        return PASS, (f"{dev}: dropped to {cold.fraction:.1%}, read "
                      f"{delta.bytes_read / MB:.0f} MB, now {warm.fraction:.1%} resident")

    # -- 4: streaming ---------------------------------------------------

    @goal(4, "NVMe streaming, bytes identical")
    def _():
        from tierinfer.storage import StorageBackend
        from tierinfer.stream import BufferPool, ExpertStreamer
        ix = need_model()
        groups = [((l, e), list(ix.expert(l, e).ranges))
                  for l in ix.moe_layers[:2] for e in range(8)]
        flat = [r for _, rs in groups for r in rs]
        slot = max(sum(r.nbytes for r in rs) for _, rs in groups)
        with StorageBackend.for_model(ix.gguf) as b:
            b.evict(flat)
            serial = hashlib.blake2b(digest_size=16)
            for _, rs in groups:
                blobs, _ = b.read(rs)
                for blob in blobs:
                    serial.update(blob)
            b.evict(flat)
            # One slot per expert: submitting all of them into a smaller pool
            # is what PoolExhausted is for, and the first version of this did
            # exactly that and called the refusal a failure.
            with ExpertStreamer(b, BufferPool(slot, len(groups)), workers=8) as s:
                t0 = time.perf_counter()
                loads = [s.submit(k, rs) for k, rs in groups]
                streamed = hashlib.blake2b(digest_size=16)
                for load in loads:
                    streamed.update(bytes(s.wait(load, timeout=120)))
                    s.release(load)
                secs = time.perf_counter() - t0
        if serial.hexdigest() != streamed.hexdigest():
            return FAIL, "streamed bytes differ from serial reads"
        mb = sum(r.nbytes for r in flat) / MB
        return PASS, f"{len(groups)} experts, {mb:.0f} MB, {mb / secs:.0f} MB/s, digests agree"

    # -- 5, 8: cache and prediction on real routing ---------------------

    @goal(8, "prediction beats the frequency floor")
    def _():
        from tierinfer.predict import (AdaptiveBlend, Frequency, Persistence,
                                       Transition, evaluate)
        from tierinfer.trace import read_trace, routings
        rows = read_trace(need_trace(), prompt=False)
        trace = list(routings(rows))
        floor = evaluate(Frequency(), trace, k=16, warmup=50).recall
        best = evaluate(AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16),
                        trace, k=16, warmup=50).recall
        if best <= floor:
            return FAIL, f"context {best:.1%} does not beat frequency {floor:.1%}"
        return PASS, (f"recall@16: frequency {floor:.1%}, adaptive {best:.1%} "
                      f"({best - floor:+.1%})")

    @goal(5, "expert cache matches LRU on real routing")
    def _():
        from tierinfer.cache import ExpertCache
        from tierinfer.tracker import ExpertTracker
        from tierinfer.trace import read_trace
        ix = need_model()
        rows = read_trace(need_trace(), prompt=False)
        tokens = [[(l, e) for l, es in r.routing.items() for e in es] for r in rows]
        size, cap = ix.expert_nbytes(), 8 * GB
        t = ExpertTracker(window=128)
        c = ExpertCache(cap, t)
        hits = misses = 0
        for token in tokens:
            t.begin_token()
            for k in token:
                if k in c:
                    hits += 1
                    c.get(k)
                else:
                    misses += 1
                    c.put(k, None, size)
            t.record(token)
        n = cap // size
        order, clock, lh, lm = {}, 0, 0, 0
        for token in tokens:
            for k in token:
                clock += 1
                if k in order:
                    lh += 1
                    order[k] = clock
                else:
                    lm += 1
                    if len(order) >= n:
                        del order[min(order, key=order.get)]
                    order[k] = clock
        got, lru = hits / (hits + misses), lh / (lh + lm)
        if got < lru - 0.01:
            return FAIL, f"{got:.1%} is below LRU's {lru:.1%}"
        return PASS, f"{got:.1%} against LRU {lru:.1%}, {c.heap_fallbacks} scan fallbacks"

    # -- 6: routing capture ---------------------------------------------

    @goal(6, "routing captured from llama.cpp")
    def _():
        from tierinfer.trace import describe
        tool = ROOT / "build" / "tierinfer-trace"
        info = describe(need_trace())
        built = "built" if tool.exists() else "not built here"
        if info.generated_tokens <= 0:
            return FAIL, "the trace has no generated tokens"
        return PASS, (f"{info.tokens} tokens, {len(info.layers)} MoE layers, "
                      f"{info.experts_seen} experts seen; tool {built}")

    # -- 7: prefetch ----------------------------------------------------

    @goal(7, "prefetch delivers under a wrong guess")
    def _():
        from tierinfer.cache import ExpertCache
        from tierinfer.index import ByteRange
        from tierinfer.predict import Prediction, Predictor
        from tierinfer.prefetch import Prefetcher
        from tierinfer.storage import StorageBackend
        from tierinfer.stream import BufferPool, ExpertStreamer
        from tierinfer.tracker import ExpertTracker
        ix = need_model()
        layer = ix.moe_layers[0]

        class Wrong(Predictor):
            name = "wrong"

            def observe(self, routing):
                pass

            def predict(self, l, sofar=None):
                return Prediction(l, (99,), (1.0,))

            def score(self, l, sofar):
                return {99: 1.0}

        slot = ix.expert(layer, 0).nbytes
        with StorageBackend.for_model(ix.gguf) as b:
            with ExpertStreamer(b, BufferPool(slot, 4), workers=2) as s:
                t = ExpertTracker()
                p = Prefetcher(s, ExpertCache(8 * slot, t), Wrong(),
                               lambda k: list(ix.expert(*k).ranges), tracker=t, depth=1)
                p.before_layer(layer, {})
                got = p.on_routing(layer, [3])
                want, _ = b.read(list(ix.expert(layer, 3).ranges))
        if got[(layer, 3)] != b"".join(want):
            return FAIL, "the fallback returned the wrong bytes"
        return PASS, f"{p.stats.stalls} stall, exact fallback returned correct bytes"

    # -- 9: VRAM --------------------------------------------------------

    @goal(9, "VRAM budget and a real transfer")
    def _():
        from tierinfer.vram import (CudaRuntime, CudaUnavailable, VramBudget,
                                    VramResidency)
        ix = need_model()
        try:
            rt = CudaRuntime()
        except CudaUnavailable as e:
            raise _Skip(str(e))
        mem = rt.memory_info()
        b = VramBudget.from_model(ix.gguf.metadata, total_bytes=mem.total,
                                  context_length=16384, layers=ix.block_count,
                                  reserve_bytes=mem.used)
        slots = min(64, b.experts(ix.expert_nbytes(), ix.always_resident_nbytes()))
        if slots <= 0:
            return FAIL, "the budget left no room for a single expert"
        host = rt.host_alloc(ix.expert_nbytes())
        ctypes.memset(ctypes.c_void_p(host), 0x5A, ix.expert_nbytes())
        try:
            with VramResidency(rt, ix.expert_nbytes(), slots) as res:
                for i in range(slots):
                    res.admit((0, i), host, ix.expert_nbytes())
                rate = res.stats.bytes_per_second / GB
        finally:
            rt.host_free(host)
        return PASS, (f"KV {b.kv_cache / GB:.2f} GB at 16k, weights "
                      f"{b.weights / GB:.1f} GB, {slots} slots at {rate:.1f} GB/s")

    # -- 10: policy -----------------------------------------------------

    @goal(10, "one policy, adapting")
    def _():
        from tierinfer.policy import TierCosts, TierPolicy
        from tierinfer.predict import AdaptiveBlend, Frequency, Persistence, Transition
        from tierinfer.tracker import ExpertTracker
        from tierinfer.trace import read_trace
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from policy import SimTier                       # noqa: E402
        rows = read_trace(need_trace(), prompt=False)[:120]
        t = ExpertTracker(window=128)
        pred = AdaptiveBlend([Frequency(), Persistence(), Transition()], k=16)
        p = TierPolicy(SimTier(2000), SimTier(6000), pred, t, TierCosts(), depth=8)
        for r in rows:
            sofar = {}
            for layer in sorted(r.routing):
                p.before_layer(layer, sofar)
                p.on_routing(layer, r.routing[layer])
                sofar[layer] = r.routing[layer]
            t.record([(l, e) for l, es in r.routing.items() for e in es])
            pred.observe(r.as_mapping())
            p.end_token()
        if p.stats.lookups == 0:
            return FAIL, "the policy resolved nothing"
        return PASS, (f"{p.stats.resident_rate:.1%} served without NVMe, "
                      f"{p.stats.seconds_per_token * 1000:.1f} ms/token, "
                      f"depth {p.depth} after {p.stats.depth_changes} moves")

    # -- 11, 12: adapters -----------------------------------------------

    @goal(11, "FreeToken adapter")
    def _():
        from tierinfer.adapters.freetoken import (FreeTokenClient, launch_arguments,
                                                  normalise, RUNTIME_FIELDS)
        from tierinfer.autoconfig import configure
        cfg = configure(need_model(), context_length=8192)
        args = launch_arguments(cfg)
        values = normalise({"throughput": {"decode_tps": 1.0}})
        if set(values) != set(RUNTIME_FIELDS):
            return FAIL, "normalised keys do not match the shared namespace"
        up = FreeTokenClient().is_up()
        return PASS, (f"{len(args) // 2} flags derived, schema agrees; "
                      f"server {'up' if up else 'not running'}")

    @goal(12, "FlowRunner capability")
    def _():
        from tierinfer.adapters.flowrunner import (CAPABILITY_NAME, CAPABILITY_VERSION,
                                                   Capability, resolve, telemetry_values)
        ix = need_model()
        cap = Capability.from_dict({"version": CAPABILITY_VERSION,
                                    "capability": CAPABILITY_NAME,
                                    "model": a.model.name, "context_length": 8192})
        r = resolve(cap, ix)
        if not r.available:
            return FAIL, "; ".join(r.refusals)
        big = resolve(Capability.from_dict({"version": CAPABILITY_VERSION,
                                            "capability": CAPABILITY_NAME,
                                            "model": a.model.name,
                                            "context_length": 131072}), ix)
        if big.available:
            return FAIL, "131k context should not resolve on this host"
        vals = telemetry_values(r.configuration)
        return PASS, (f"8k resolves ({vals['capability.vram_experts']} VRAM experts), "
                      "131k refused as it should be")

    # -- 13: telemetry --------------------------------------------------

    @goal(13, "telemetry round-trips")
    def _():
        from tierinfer.telemetry import Telemetry, deltas, read
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            with Telemetry(path, run="smoke") as t:
                t.open_run(model=a.model.name)
                for hits in (0, 10):
                    t.snapshot(values={"runtime.decode_tps": float(hits)})
                t.close_run()
            recs = read(path)
            diffs = list(deltas(recs))
        if len(recs) != 4 or not diffs or diffs[0]["runtime.decode_tps"] != 10.0:
            return FAIL, f"{len(recs)} records, {len(diffs)} deltas"
        return PASS, f"{len(recs)} records, deltas correct, schema v1"

    # -- 14: autoconfig -------------------------------------------------

    @goal(14, "configuration derived and refused")
    def _():
        from tierinfer.autoconfig import Host, configure
        ix = need_model()
        host = Host.measure()
        ok = configure(ix, context_length=8192, host=host)
        if not ok.usable:
            return FAIL, "; ".join(ok.problems)
        if host.has_gpu:
            bad = configure(ix, context_length=131072, host=host)
            if bad.usable:
                return FAIL, "131k context should have been refused"
        return PASS, (f"{ok.vram_experts} VRAM + {ok.ram_experts} RAM experts, "
                      f"{ok.resident_share:.0%} of routed weights resident")

    # -- 15: failure safety ---------------------------------------------

    @goal(15, "a wrong guess never returns wrong bytes")
    def _():
        from tierinfer.safety import GuardStats, audit, exact_load, guard
        from tierinfer.storage import StorageBackend
        ix = need_model()
        findings = audit()
        if findings:
            return FAIL, "; ".join(str(f) for f in findings)
        layer = ix.moe_layers[0]
        ranges = lambda k: list(ix.expert(*k).ranges)
        with StorageBackend.for_model(ix.gguf) as b:
            want = exact_load(b, ranges((layer, 5)))
            stats = GuardStats()
            for broken in (lambda k: None,
                           lambda k: b"short",
                           lambda k: (_ for _ in ()).throw(RuntimeError("boom"))):
                if guard(broken, b, ranges, stats)((layer, 5)) != want:
                    return FAIL, "a guarded load returned the wrong bytes"
        return PASS, (f"audit clean, {stats.fallbacks} fallbacks over "
                      f"{len(stats.reasons)} reasons, bytes identical")

    # -- verdict --------------------------------------------------------

    failed = [r for r in results if r[2] == FAIL]
    skipped = [r for r in results if r[2] == SKIP]
    print(f"\n{len(results) - len(failed) - len(skipped)} passed, "
          f"{len(skipped)} skipped, {len(failed)} failed")
    for n, name, _, detail in failed:
        print(f"  FAILED {n}. {name}: {detail}")
    for n, name, _, detail in skipped:
        print(f"  skipped {n}. {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

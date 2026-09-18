# SCOPE — TierInfer

> The Human's full scope is `docs/SCOPE-ORIGINAL.md`; this file is the
> working condensation with per-goal status. `docs/SCOPE-ADDENDUM-480B.md`
> extends it, `docs/AUDIT-2026-09-18.md` measured the status below against
> the code, and `docs/CHECKPOINTS.md` is the durable trail. Where a goal's
> status here was rewritten by the audit, the rewrite says so.

## Purpose

Run models whose weights exceed practical VRAM and RAM by treating NVMe as a
deliberate third inference memory tier, and by moving weights between tiers
at the granularity the model actually uses — for MoE, the expert.

## The measurement that motivates it

Taken on this workstation before the project existed. RTX 5090 32 GB, 187 GB
RAM, Samsung 990 PRO, llama.cpp `-ngl 25`, GLM-4.5-Air IQ4_XS (~57 GB):

| Condition | Decode | Storage behaviour |
|---|---:|---|
| Unconstrained RAM | 7.7 tok/s | idle after load |
| `MemoryHigh` 48 GB / `MemoryMax` 52 GB | 5.5 tok/s | 500–830 MB/s, ~200 000 reads/s, ~4 KB each |

Two conclusions. NVMe can back inference. And generic demand paging does it
by reacting to a page that is already missing, in the smallest possible unit.

## The target to beat

Not "the model runs". Under equivalent memory constraint, against 5.5 tok/s:
fewer storage operations, larger effective reads, fewer synchronous stalls,
and higher decode throughput. Improvement must be measured against the
constrained baseline, not against the unconstrained one.

## Order of work

Each step must be measurable on its own before the next begins.

1. **Repository and foundation.** — *done*
2. **Model layout indexing.** GGUF directory, tensor classification, expert
   byte ranges. — *done for `glm4moe` and, since 2026-09-18, for `qwen3moe`
   and for split models: a `gguf-split` set is read as one model, every
   byte range names its shard, and the storage layer holds one descriptor
   per shard. Before that a six-shard model indexed as its first shard.*
3. **Reproducible baseline benchmark.** Automate the manual experiment:
   warm cache, cold cache, RAM-constrained, and later TierInfer-enabled, with
   throughput, NVMe bandwidth, read-size distribution, IOPS and residency.
   — *done for the four conditions that exist before TierInfer; results and
   method in `benchmarks/BASELINE.md`. The headline: 8 tokens touch ~3.2 GB of
   experts and read all 56.5 GB, and a 32 GB ceiling costs 7.4x throughput
   while reading 113.6 GB — twice the model. Read-size distribution is
   reported as a mean only; a histogram needs blktrace, which needs root.*
4. **Explicit NVMe→RAM expert streaming.** Read a named expert into a
   controlled buffer, asynchronously, independently of demand paging, and
   time it. — *built: `pread` on a plain fd (nothing mapped), worker threads,
   a pool of fixed slots that raises rather than growing, per-load queue and
   read timing, and a synchronous `load_now` that consults nothing — the
   fallback goal 15 requires. Measured on the model
   (`benchmarks/STREAMING.md`): 3.47x demand paging at eight workers, and
   3.0 GB/s against llama.cpp's own 1.54 GB/s cold load, with every method
   returning byte-identical data. Concurrency is the whole difference — a
   single-threaded pread is slower than faulting.* *480B addendum (2026-09-18):
   on the USB RAID0 the 480B sits on, eight concurrent expert reads measured
   0.95× a serial one — the pipe fills at ~0.75 GB/s for expert-sized random
   reads either way — so concurrency is a device property and
   `autoconfig.probe_concurrency` now measures it rather than the code
   assuming it (`benchmarks/480b/VALIDATION-480B.md` §4.3).*
5. **Bounded RAM expert cache.** Insertion, hit, miss, eviction, pinning,
   adaptive retention. — *done and measured: +1.5 to +13.1 points of hit rate
   over LRU (widest where the cache is smallest), and victim selection is a
   lazy min-heap with revalidation, flat at 8 µs from 1 000 to 6 000
   residents against the earlier scan's 354 to 1 031 µs, costing 0.1 point of
   hit rate; every fall back to the exact scan is counted in `heap_fallbacks`*
   *480B addendum: holding the real bytes under real routing on the 480B,
   86.8 % hit at 100 GiB and 91.7 % at 150 GiB, reading 36–65 % fewer bytes in
   20–40× fewer operations than Linux demand paging with more RAM
   (`VALIDATION-480B.md` §4). Feeding the reload-cost term with measured
   per-load times made it evict by queueing noise (−3 points); the term is
   constant until a per-expert cost exists.*
6. **llama.cpp integration.** Real inference with TierInfer assisting
   residency, benchmarkable against native mmap. A clean baseline mode must
   remain. — *routing capture done and nothing in llama.cpp is patched:
   `tools/trace` reads the `ffn_moe_topk-<layer>` tensors through the public
   `cb_eval` hook and links against the existing build, so the clean baseline
   is the default rather than a mode. Two 400-token traces from GLM-4.5-Air
   are in `traces/`, and `benchmarks/REAL-ROUTING.md` reports what they
   overturn. Residency assist is measured and settled: it cannot be
   done advisorily. WILLNEED is a no-op under a full ceiling and DONTNEED
   cannot evict pages a live mapping holds, shown end to end and in isolation;
   the assist flags were removed from the tool on 2026-09-18 because they
   could not do what they were named for.*

   ***Status corrected by the 2026-09-18 audit: observability only. Goal 6 as
   the original scope words it — real inference while TierInfer controls or
   assists weight residency, benchmarkable against native mmap — is NOT
   met.*** *No loader exists; in every inference this project has run, all
   weight I/O was Linux demand paging. The integration point is
   `ggml_mul_mat_id` reading expert slabs out of the mapped tensor
   (`docs/AUDIT-2026-09-18.md`, item 12); owning it is the remaining work,
   and it is the largest piece left in the project.*
7. **Async prefetch.** Read-ahead overlapping compute; measure stalls
   eliminated, accuracy, lead time and wasted bandwidth. — *built: the one
   place a guess causes I/O, and the only place the safety rule has teeth —
   a routed expert nobody anticipated gets an exact synchronous read, and is
   counted as a stall. All four numbers reported — plus `late`, split from
   `useful` by whether the load had finished when the routing asked.*
   *480B addendum: measured end to end under replayed real routing (no
   compute — goal 6): prefetch at depth 8 lands within 1 % of cache-only on
   every I/O figure; without compute to overlap, two of three speculative
   reads are late or wasted, and depth 16 doubles the waste for nothing. On
   this model the predictor names experts the cache already holds. A
   negative result (`VALIDATION-480B.md` §4).*
8. **Expert prediction and prerouter.** Ranked prediction of upcoming
   experts, evaluated against actual routing. — *four predictors (frequency,
   persistence, transition, adaptive blend) and a recall@k harness that never
   shows a layer its own routing. Measured on a synthetic trace: context beats
   the frequency floor by about 1 point, and the adaptive blend backs whichever
   part is measurably winning. That result is about the code, not the model —
   `evaluate` takes any iterable of routings, so a captured trace scores through
   the same harness once goal 6 can produce one. On real GLM routing the
   adaptive blend beats the frequency floor by 13.9 points at k=16.*
   ***Audit 2026-09-18: the predictors are heuristic. The trainable prerouter
   the original scope requires is NOT implemented.***
9. **VRAM working set.** Explicit GPU residency inside a budget that leaves
   room for KV cache, activations and workspace. — *done and measured
   (`benchmarks/VRAM.md`). The budget is derived from the model's metadata,
   and its KV term is identical to llama.cpp's own allocation at three
   contexts; the runtime overhead is measured rather than derived, because it
   belongs to the runtime. On real routing a 32 GB card holds 22–25 GB of
   experts at a 76–79% hit rate, and the misses cost about a fifth of a warm
   token — against 7.4x for the same model under a RAM ceiling. Below one
   token's working set the hit rate is not low but zero, which is what
   131k context does.*
10. **Adaptive tier policy.** One policy over the runtime signals, adapting
    during inference. — ***Status corrected by the 2026-09-18 audit:
    SIMULATED, not done.*** *`tierinfer.policy` is a cost model over the
    measured tier constants, applied to real routing (`benchmarks/POLICY.md`,
    whose last section already said so). It has never run over the real
    tiers: it binds to a `SimTier` interface that neither `ExpertCache` nor
    `VramResidency` implements, and the seconds it reports are computed, not
    observed — they are exported under the `sim.` telemetry namespace for
    that reason. What the simulation says: tiering is worth 17x (1 318
    ms/token to 75), the dial on top is worth 1%, the adaptive arm reaches
    the best fixed depth unaided, and a 3.8% miss rate costs two thirds of
    the time. A policy that moves bytes is the loader's work (goal 6).*
11. **FreeToken adapter.** — ***partial (audit 2026-09-18): configuration in
    and counters out, verified against FreeToken's own argument parser; the
    NVMe tier beneath FreeToken, the intake of its routing and the
    real-inference benchmark the original scope asks for do not exist.***
    *FreeToken already tiers experts itself
    and owns its loading path, and goal 6 measured what happens to an adapter
    that tries to manage residency in a runtime that owns its own: nothing,
    slowly. So the adapter does the two things that are left. Configuration
    in: the derived budget becomes `--moe-cache-size`, `--moe-cache-policy`
    and `--kv-reserve-tokens`, and an unusable configuration is refused
    before a server spends forty seconds loading 56 GB to discover it.
    Telemetry out: `/v1/stats` is normalised into the shared `runtime.*`
    namespace, with an unreported counter coming back absent rather than
    zero. Standard-library HTTP only.*
12. **FlowRunner capability.** Declarative configuration, telemetry out.
    — ***schema done, DISCONNECTED (audit 2026-09-18): nothing in FlowRunner
    reads it, so a flow cannot select TierInfer today.*** *A flow declares
    what the work needs — a model, a context, and
    optionally the residency share below which the step should be refused
    rather than run slowly and believed — and is deliberately unable to
    declare a VRAM size, because that belongs to whatever machine the flow
    lands on. `resolve` puts the declaration and the host together and
    produces a configuration or a refusal naming what is missing. Unknown
    fields are refused rather than ignored. Telemetry goes out in the shared
    schema.*
13. **Unified telemetry.** Runtime-independent schema, machine-readable.
    — *done. Two record kinds, events and snapshots, versioned, JSON Lines,
    line-buffered so a reader arriving mid-run sees everything so far. Field
    lists live in one place so adding a counter cannot silently widen the
    schema, a missing counter is an error rather than a gap, and a file that
    does not open with `run.open` is refused because its records cannot be
    attributed. Nothing is derived on the way out — rates belong to the
    reader, who can then check them.*
14. **Automatic configuration.** Derive budgets from host and model. —
    *done. VRAM from the driver, RAM from /proc/meminfo, the model's shape
    from its GGUF, the tier constants from the benchmarks. Every choice is
    reported as a decision, and a figure that cannot be measured here is
    absent rather than defaulted. It refuses rather than returning numbers
    that cannot work: a context whose KV cache will not fit, a host that
    cannot hold one expert, or a VRAM budget below one token's working set —
    goal 9's zero-hit-rate case, which is why 131k context is refused on this
    host.* *480B addendum: derives correctly for the split `qwen3moe` model
    (681 VRAM experts at 4k, 131k refused), and `configure_measured` adds the
    one figure that needs the device touched — whether concurrent demand
    reads pay — from a one-second probe on the model's own files.*
15. **Failure safety.** A prediction miss falls back to an exact load.
    Always. — *done, and stated in one place rather than left distributed
    across the modules that happen to honour it. `exact_load` consults
    nothing; `guard` turns every way a speculative loader can be wrong —
    absent, raising, too few bytes, too many — into an exact load, counting
    each reason apart. A failure of the exact path itself is fatal, because
    inventing a fallback below it would mean returning weights that are not
    the model's. `audit` reports any module using a predictor without a
    reachable exact path, which is how the invariant would actually be lost:
    by omission when a new speculative path is added.*

## Rules the implementation is held to

**Correctness outranks throughput.** Prediction decides what is prefetched.
It never decides what is executed. Exact routing and exact loading stay
reachable from every path.

**Refuse rather than guess.** A tensor whose layout does not divide as
expected produces an error, not an approximate byte range. Reading the wrong
weights silently is the one failure this project cannot have.

**The core stays runtime-independent.** Policy, metrics and tier management
do not live inside a fork of an inference engine. Adapters may patch; the
core may not depend on one.

**Measure, do not assume.** Drive characteristics, expert sizes and model
topology are read from the host and the file at run time. No hardcoded
bandwidth, no assumed expert count, no fixed layout.

**Every mode stays available.** Baseline, observability, cache, prefetch and
full adaptive are diagnostic instruments, not stages to be discarded once the
last one works.

## Boundaries

TierInfer does not replace llama.cpp, FreeToken, FlowRunner, CUDA or a model
format. It is the memory and streaming layer between an inference runtime and
the machine's memory hierarchy, and it must remain useful without FlowRunner.

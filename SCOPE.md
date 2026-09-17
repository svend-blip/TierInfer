# SCOPE — TierInfer

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
   byte ranges. — *done for `glm4moe`; other architectures untested*
3. **Reproducible baseline benchmark.** Automate the manual experiment:
   warm cache, cold cache, RAM-constrained, and later TierInfer-enabled, with
   throughput, NVMe bandwidth, read-size distribution, IOPS and residency.
4. **Explicit NVMe→RAM expert streaming.** Read a named expert into a
   controlled buffer, asynchronously, independently of demand paging, and
   time it.
5. **Bounded RAM expert cache.** Insertion, hit, miss, eviction, pinning,
   adaptive retention. — *policy done and measured (+7 to +14 points of hit
   rate over LRU); victim selection is an O(n) scan costing over a
   millisecond at 3 000 resident experts and must be replaced by a heap
   before this goes near an inference loop*
6. **llama.cpp integration.** Real inference with TierInfer assisting
   residency, benchmarkable against native mmap. A clean baseline mode must
   remain.
7. **Async prefetch.** Read-ahead overlapping compute; measure stalls
   eliminated, accuracy, lead time and wasted bandwidth.
8. **Expert prediction and prerouter.** Ranked prediction of upcoming
   experts, evaluated against actual routing.
9. **VRAM working set.** Explicit GPU residency inside a budget that leaves
   room for KV cache, activations and workspace.
10. **Adaptive tier policy.** One policy over the runtime signals, adapting
    during inference.
11. **FreeToken adapter.**
12. **FlowRunner capability.** Declarative configuration, telemetry out.
13. **Unified telemetry.** Runtime-independent schema, machine-readable.
14. **Automatic configuration.** Derive budgets from host and model.
15. **Failure safety.** A prediction miss falls back to an exact load. Always.

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

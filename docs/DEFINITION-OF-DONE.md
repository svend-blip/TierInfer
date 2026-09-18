# TierInfer — Consolidated Definition of Done

> Recorded verbatim from the Human on 2026-09-18. This is the single
> authoritative end-state; `SCOPE.md`, `docs/SCOPE-ORIGINAL.md` and the
> addenda determine implementation details and sequencing, this document
> determines whether the project has reached its objective.

## Purpose

This document defines the single authoritative end-state for TierInfer.

Fable 5 must use this Definition of Done as the final target when planning, implementing, reviewing, testing, reconciling scope state, and deciding whether TierInfer is complete.

TierInfer is not complete because architecture exists, source files exist, tests pass, interfaces have been created, or individual scope goals have previously been marked complete.

TierInfer is complete only when the entire system demonstrably provides a real, correct, measurable, and production-connected adaptive inference memory hierarchy:

```text
Tier 0 — GPU VRAM
        ↕
Tier 1 — System RAM
        ↕
Tier 2 — NVMe storage
```

for supported large local models, including sparse MoE workloads, with working integration for both:

```text
llama.cpp
FreeToken
```

The system must allow models whose useful working set or total weights exceed available VRAM and, where applicable, exceed available VRAM + RAM, while intelligently using NVMe as an explicit inference tier rather than merely depending on uncontrolled operating-system paging.

Correctness is mandatory.

Performance improvement must be measured rather than assumed.

All claims of completion must be supported by executable production paths and observed runtime evidence.

---

# 1. Core End State

At Definition of Done, TierInfer must provide a coherent runtime system capable of managing model data across:

```text
VRAM
RAM
NVMe
```

according to actual model activity and available resources.

The hierarchy must be explicit and observable.

TierInfer must know:

* what model data exists;
* where relevant model data resides;
* what data is currently needed;
* what data is likely to be needed next;
* what should remain hot;
* what can be evicted;
* what should be prefetched;
* what storage ranges correspond to model structures;
* what transfers are in flight;
* what transfers succeeded or failed;
* what cache/prefetch decisions helped or hurt inference.

TierInfer must not merely rename Linux mmap/page-cache behavior as TierInfer tiering.

---

# 2. Runtime Support

TierInfer must support both of the following inference-runtime integrations as first-class production paths:

```text
TierInfer
   │
   ├── llama.cpp adapter/integration
   │
   └── FreeToken adapter/integration
```

Both integrations must use the common TierInfer architecture wherever technically applicable.

TierInfer must not become two unrelated implementations that happen to share a project name.

Runtime-specific adapters may expose different capabilities, but shared policy, telemetry, model-layout concepts, tier-management concepts, and correctness semantics should remain common.

---

# 3. llama.cpp Definition of Done

The llama.cpp integration is complete only when TierInfer can participate in real llama.cpp inference.

It must be possible to run a supported GGUF model through a documented TierInfer-enabled llama.cpp execution path.

The integration must demonstrate real model execution and real TierInfer activity.

Where applicable, TierInfer must be capable of observing or controlling:

```text
model layout
GGUF shards
tensor/range locations
expert-related ranges
RAM residency
VRAM residency
NVMe reads
cache decisions
prefetch decisions
evictions
fallback reads
```

The integration must not consist only of:

* wrappers around llama.cpp commands;
* log parsing;
* benchmark observation;
* configuration generation;
* simulated data movement.

TierInfer must materially participate in the inference data path where the effective architecture requires it.

Native llama.cpp behavior must remain available as a baseline and fallback where appropriate.

---

# 4. FreeToken Definition of Done

FreeToken must be a first-class TierInfer runtime target.

TierInfer must integrate with real FreeToken inference rather than merely being able to launch a FreeToken process.

The integration must determine which TierInfer mechanisms can be applied directly to FreeToken's execution architecture and implement the necessary adapter/integration layer.

Where FreeToken already provides mechanisms for:

```text
GPU residency
host-memory offload
expert management
prefetch
cache management
MoE execution
```

TierInfer must integrate with those mechanisms rather than blindly duplicating or fighting them.

TierInfer must add value through the common adaptive hierarchy, policy, observability, storage tier, prediction, and/or orchestration mechanisms defined by the effective TierInfer scope.

A real FreeToken inference workload must demonstrate that the TierInfer integration is active.

TierInfer telemetry must be able to distinguish:

```text
FreeToken-native behavior
TierInfer decisions
operating-system behavior
```

as far as technically practical.

A FreeToken adapter that exists but is not connected to a real inference execution path does not satisfy this Definition of Done.

---

# 5. Model Layout Awareness

TierInfer must contain a functioning Model Layout Inspector.

For supported model formats/runtimes it must identify enough of the model's physical and logical layout to make meaningful tiering decisions.

For GGUF models this includes, where applicable:

```text
shards
tensor metadata
tensor byte ranges
tensor offsets
model layers
MoE structures
expert-related tensors/ranges
shared structures
```

The inspector must correctly handle sharded models.

Byte offsets and ranges must correspond to actual model data.

TierInfer must never fabricate mappings that cannot be established reliably.

Unsupported structures must be reported explicitly.

---

# 6. Explicit NVMe Tier

NVMe must function as an intentional Tier 2 inference resource.

TierInfer must be capable of requesting model data from storage using model-aware ranges rather than relying exclusively on reactive 4 KiB page faults.

The storage backend must support efficient access patterns appropriate for the hardware.

Where beneficial, TierInfer must be able to:

```text
combine adjacent ranges
batch reads
issue larger reads
perform asynchronous reads
maintain multiple reads in flight
prefetch expected data
overlap storage I/O with computation
```

TierInfer must measure the resulting physical behavior rather than assume these techniques are effective.

---

# 7. RAM Expert/Data Cache

TierInfer must provide a bounded RAM cache for model data that benefits from reuse.

For sparse MoE workloads, expert weights must be cacheable where technically possible.

The cache must implement real:

```text
admission
lookup
hit
miss
residency
eviction
capacity accounting
concurrency control
```

Cache capacity must respect configured and automatically determined memory budgets.

TierInfer must not allow the cache to grow without bound.

Cache statistics must correspond to real accesses.

---

# 8. VRAM Working-Set Management

TierInfer must support a bounded VRAM working set where the runtime integration permits it.

The VRAM tier should preferentially contain model data that provides the highest practical inference value.

For sparse MoE models this may include hot experts or predicted near-future experts.

TierInfer must track:

```text
VRAM residency
promotion
reuse
eviction
working-set size
hits
misses
transfer cost
```

The system must preserve enough VRAM for other runtime requirements, including KV cache and runtime overhead.

TierInfer must not maximize weight residency at the expense of making inference unstable.

---

# 9. Expert Activity Tracking

For supported sparse MoE workloads, TierInfer must observe actual expert activity where the runtime exposes or can reasonably expose it.

The tracker must maintain useful information such as:

```text
expert activation frequency
recent activation
reuse patterns
hot experts
cold experts
transition patterns
```

The data must originate from real inference activity.

File-access patterns alone must not be falsely reported as authoritative expert-routing information.

---

# 10. Expert Prediction and Prerouting

TierInfer must implement the expert-prediction/prerouter functionality required by the effective scope.

The mechanism must be connected to real inference.

Predictions must be capable of influencing prefetch/residency decisions.

The system must measure:

```text
predictions issued
correct predictions
incorrect predictions
useful predicted prefetches
late prefetches
unused prefetches
prediction precision
prediction recall where meaningful
```

A predictor that returns synthetic values, constant predictions, or results that never influence execution is not implemented.

If trainable prerouting is part of the effective implementation, its training, persistence, loading, and inference paths must all function.

---

# 11. Asynchronous Prefetch

TierInfer must contain a real asynchronous prefetch engine.

Prefetch must occur early enough that storage access can overlap useful computation where the workload permits.

The engine must support:

```text
request scheduling
bounded concurrency
deduplication
cancellation or supersession where appropriate
completion tracking
error handling
usefulness tracking
```

Telemetry must distinguish:

```text
prefetch issued
prefetch completed
prefetch useful
prefetch late
prefetch unused
demand read
```

Issuing a read before another read does not automatically qualify as successful prefetching.

---

# 12. Adaptive Tier Manager

The Tier Manager must make real residency and movement decisions.

Its decisions must consider relevant factors such as:

```text
available VRAM
available RAM
model size
active working set
expert activity
cache pressure
storage performance
transfer cost
prefetch effectiveness
runtime requirements
```

TierInfer must be capable of adapting when workload behavior changes.

The hierarchy must not depend solely on static manually selected thresholds.

Manual overrides may exist, but automatic configuration must provide a safe and useful default.

---

# 13. Automatic Configuration

TierInfer must inspect the host and derive a usable configuration.

At minimum, relevant detection should include:

```text
available VRAM
available system RAM
model size
model layout
storage characteristics
runtime capabilities
```

The resulting configuration must preserve safety margins.

A user should not need to manually calculate every VRAM, RAM, cache, and storage threshold to obtain a valid TierInfer configuration.

Advanced manual overrides may remain available.

---

# 14. Unified Telemetry

TierInfer must provide trustworthy unified telemetry.

At minimum, where technically available, telemetry must cover:

```text
VRAM usage
RAM usage
NVMe activity

Tier 0 hits/misses
Tier 1 hits/misses
Tier 2 reads

promotions
demotions
evictions

bytes transferred
transfer latency

prefetch issued
prefetch useful
prefetch late
prefetch wasted

expert activity
prediction activity

storage stall time
inference throughput
time to first token
```

Telemetry must describe what actually happened.

No metric may report intended policy as observed behavior.

---

# 15. Correctness

TierInfer must never trade model correctness for apparent performance without explicitly operating in a documented approximate mode.

Normal TierInfer operation must preserve inference correctness.

Data movement must not:

* load incorrect tensor ranges;
* mix shards;
* corrupt cached data;
* use stale buffers;
* race eviction against execution;
* expose partially loaded data;
* silently lose failed I/O.

Concurrency and asynchronous I/O paths must be tested for correctness.

---

# 16. Graceful Degradation

TierInfer must remain safe when the ideal fast path is unavailable.

Examples include:

```text
VRAM exhaustion
RAM pressure
cache exhaustion
prefetch arriving late
incorrect prediction
NVMe latency spike
I/O failure
unsupported model structure
unsupported runtime capability
```

TierInfer must either:

```text
fall back to a correct slower path
```

or fail explicitly with actionable diagnostics.

Silent corruption is never acceptable.

Silent fallback that makes telemetry claim TierInfer optimization is active when it is not is also unacceptable.

---

# 17. Quality Requirements

Before final completion, the repository must contain no material implementation defects hidden behind architectural scaffolding.

The final quality review must search for and resolve relevant:

```text
TODO
FIXME
stub
placeholder
fake implementation
synthetic production telemetry
dead production path
disconnected component
ignored configuration
unused integration
silent error
unsafe fallback
resource leak
race condition
incorrect accounting
```

Not every comment containing TODO must necessarily disappear.

The requirement is that no unresolved placeholder or incomplete mechanism may remain for functionality required by the effective TierInfer scope.

---

# 18. Tests

TierInfer must have automated tests covering the important correctness logic.

Tests must include appropriate combinations of:

```text
unit tests
integration tests
runtime-adapter tests
cache tests
layout tests
range tests
tier-policy tests
failure tests
concurrency tests
```

Mocks may be used where appropriate.

However, a mocked test cannot be the sole evidence that a production integration works.

Real-runtime validation is required for both llama.cpp and FreeToken.

---

# 19. Large-Model System Validation

TierInfer must be validated against a model large enough to create genuine memory-tier pressure.

The current designated large-model validation target is:

```text
Qwen3-Coder-480B-A35B-Instruct
Q4_K_M GGUF
6 shards
~290 GB decimal / ~270 GiB
```

Stored at:

```text
/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/
```

This model must be used to validate the llama.cpp TierInfer path unless a verified technical incompatibility makes that impossible.

The system under test contains approximately:

```text
RTX 5090
32 GB VRAM

System RAM
192 GB

Tier 2 storage
OWC Express 4M2 RAID0
/data/ai-data
```

The test must demonstrate genuine pressure across the memory hierarchy.

---

# 20. Native Baselines

TierInfer performance must be compared against the corresponding runtime without TierInfer optimization.

For llama.cpp:

```text
native llama.cpp
versus
llama.cpp + TierInfer
```

For FreeToken:

```text
native FreeToken
versus
FreeToken + TierInfer
```

Comparisons must be controlled and reproducible.

TierInfer must not claim improvements based solely on comparisons against theoretical storage behavior.

The native operating system, runtime, filesystem, RAID, mmap, page cache, runtime-native offload, and caching mechanisms already provide optimizations.

TierInfer must be evaluated against those real baselines.

---

# 21. Performance Evidence

Performance claims require repeated measurements.

Relevant metrics include:

```text
prompt tokens/s
generation tokens/s
time to first token
NVMe bytes/token
NVMe IOPS
average read size
storage latency
storage stall time
RAM cache hit rate
VRAM working-set hit rate
prefetch usefulness
prediction effectiveness
CPU overhead
```

Separate cold and warm cache behavior where relevant.

Use repeated runs and report representative statistics rather than selecting the best single run.

A negative result must be preserved as evidence.

TierInfer is not allowed to hide regressions.

---

# 22. Performance Success Criterion

TierInfer does not need to prove that NVMe is as fast as RAM or VRAM.

That is not the objective.

TierInfer must demonstrate that intelligently managed storage-backed inference provides a meaningful advantage over the corresponding unmanaged/native fallback for workloads where model data cannot remain in the faster tiers.

The advantage may be demonstrated through one or more measurable improvements such as:

```text
higher generation throughput
lower storage stall time
lower NVMe bytes/token
lower IOPS
larger/coalesced useful reads
higher effective cache reuse
better overlap of I/O and compute
more stable latency
ability to execute a larger model within bounded RAM
```

Performance conclusions must be evidence-based.

If the current TierInfer strategy does not outperform the native baseline, the result must be diagnosed and the implementation improved within the effective scope rather than declaring success.

---

# 23. FlowRunner Compatibility

TierInfer must remain usable by the broader runtime ecosystem.

FlowRunner must be able to select/use TierInfer-capable inference without needing to understand the internal implementation of:

```text
VRAM cache
RAM cache
NVMe streaming
expert prediction
prefetch
```

The intended abstraction is:

```text
FlowRunner
     │
     ▼
Inference Runtime / TierInfer integration
     │
     ├── llama.cpp
     │
     └── FreeToken
```

FlowRunner should consume a stable runtime capability rather than duplicate TierInfer policy.

---

# 24. Reproducibility

Every important benchmark and system validation must record enough information to reproduce it.

At minimum:

```text
TierInfer revision
runtime revision/version
model
quantization
model path
context size
runtime arguments
TierInfer configuration
VRAM state
RAM state
storage location
kernel
GPU driver
CUDA version where relevant
test prompt/workload
cache state
measurement commands
results
```

---

# 25. Durable State and Cold Resume

TierInfer development must remain compatible with the existing durable scope/checkpoint workflow.

Fable 5 must record:

```text
goals
decisions
blockers
coverage
quality findings
benchmark evidence
checkpoints
next action
```

A new process must be able to recover the effective scope and continue without relying on previous conversational context.

Durable state must agree with repository reality.

---

# 26. Documentation

At completion, documentation must explain how to:

```text
install TierInfer
validate the installation
inspect detected hardware
inspect a model
configure tiers
run with llama.cpp
run with FreeToken
run through FlowRunner where applicable
observe telemetry
benchmark native vs TierInfer
diagnose failures
disable TierInfer/fall back safely
```

Documentation must describe the actual implementation.

Do not document planned behavior as if it already exists.

---

# 27. Explicit Non-Completion Conditions

TierInfer is NOT done if any of the following is true:

* llama.cpp support exists only as a stub, wrapper, mock, or disconnected adapter;
* FreeToken support exists only as a stub, wrapper, mock, or disconnected adapter;
* NVMe usage is merely Linux paging presented as TierInfer behavior;
* RAM caching is simulated;
* VRAM working-set telemetry is synthetic;
* expert activity is fabricated or inferred without a valid mapping;
* predictor output does not affect execution;
* prefetch is synchronous while being reported as asynchronous;
* cache hits are not based on actual data reuse;
* telemetry reports desired state instead of observed state;
* large-model validation has not been performed;
* only mocked runtime tests exist;
* significant scope-required stubs remain;
* runtime failures can cause silent incorrect inference;
* native-vs-TierInfer comparison has not been performed;
* previously completed goals contradict current repository reality;
* documentation claims functionality that production code does not provide.

---

# 28. Final Definition of Done

TierInfer is DONE only when all of the following are simultaneously true:

1. The repository has passed a thorough implementation-quality audit.
2. All functionality required by the effective TierInfer scope is implemented in real production paths rather than stubs, mocks, placeholders, or disconnected abstractions.
3. TierInfer provides an explicit and observable `VRAM ↔ RAM ↔ NVMe` inference hierarchy.
4. Model layout and relevant storage ranges can be identified correctly.
5. NVMe functions as a deliberate model-aware Tier 2 resource.
6. RAM caching is real, bounded, concurrent-safe, measurable, and used by inference.
7. VRAM working-set management is real and measurable where supported by the runtime.
8. Sparse-MoE expert activity is tracked from valid runtime/model information where supported.
9. Expert prediction/prerouting is real, connected to execution, and measurable as required by scope.
10. Asynchronous prefetch is real, bounded, overlapped with useful work where possible, and its usefulness is measured.
11. Tier policy adapts to model, hardware, workload, and resource pressure.
12. Automatic configuration produces a safe usable configuration without requiring the user to manually calculate the entire memory hierarchy.
13. Telemetry accurately describes real transfers, residency, cache behavior, predictions, prefetches, failures, and inference performance.
14. llama.cpp is a fully functioning TierInfer runtime integration demonstrated using real inference.
15. FreeToken is a fully functioning TierInfer runtime integration demonstrated using real inference.
16. Runtime-specific native mechanisms are respected and integrated rather than unnecessarily duplicated.
17. Native llama.cpp versus TierInfer-enabled llama.cpp has been benchmarked reproducibly.
18. Native FreeToken versus TierInfer-enabled FreeToken has been benchmarked reproducibly.
19. The downloaded Qwen3-Coder-480B-A35B Q4_K_M model has been used for genuine large-model validation of the llama.cpp path unless a documented fundamental runtime incompatibility prevents it.
20. Large-model testing demonstrates genuine memory pressure in which Tier 2 participates in active inference.
21. TierInfer behavior has been distinguished from Linux mmap/page-cache and RAID behavior sufficiently to substantiate TierInfer's contribution.
22. Correctness is maintained during promotion, eviction, prefetch, cache reuse, and concurrent I/O.
23. Resource exhaustion and I/O failures either degrade to a correct path or produce explicit actionable failure.
24. Automated correctness and integration tests pass.
25. Real llama.cpp runtime validation passes.
26. Real FreeToken runtime validation passes.
27. Repeated benchmarks provide trustworthy evidence about TierInfer's performance and overhead.
28. Performance regressions and unsuccessful optimization strategies are reported rather than hidden.
29. FlowRunner can consume TierInfer-enabled llama.cpp and FreeToken capabilities without reimplementing TierInfer's internal memory-management policy.
30. Documentation corresponds to the actual production implementation.
31. Durable scope state, tests, documentation, telemetry, and repository implementation agree.
32. No scope-required mechanism remains materially stubbed, simulated, disconnected, or falsely reported as operational.

---

# 29. Final System Target

```text
                         FlowRunner
                             │
                             ▼
                    ┌─────────────────┐
                    │    TierInfer    │
                    │ adaptive policy │
                    │ model layout    │
                    │ expert activity │
                    │ prediction      │
                    │ prefetch        │
                    │ telemetry       │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
             llama.cpp                FreeToken
                 │                       │
                 └───────────┬───────────┘
                             ▼
                  Tier 0  VRAM  hot working set
                             ↕
                  Tier 1  RAM   expert/data cache
                             ↕
                  Tier 2  NVMe  model-aware storage, async/prefetch I/O
```

The important property is not the diagram. The important property is that
every relevant arrow in the diagram corresponds to real, tested, observable
production behavior.

---

# 30. Instruction to Fable 5

Treat this document as the consolidated end-state Definition of Done for TierInfer.

Use the existing effective SCOPE and addenda to determine implementation details and work sequencing, but use this Definition of Done to determine whether the project has actually reached its objective.

Do not optimize for completing checkboxes. Do not preserve a completed status when repository evidence contradicts it. Do not accept architectural scaffolding as implementation. Do not accept mocks as proof of runtime integration. Do not accept Linux paging as proof of TierInfer NVMe management. Do not accept a runtime adapter that only launches the runtime as proof of TierInfer integration. Do not accept telemetry without evidence that the measured event occurred. Do not accept a single successful benchmark as proof of a performance improvement.

Continue implementation, correction, integration, validation, and reconciliation until the production system itself satisfies this Definition of Done.

The final question is not: "Has the TierInfer scope been implemented?"

The final question is: "Can TierInfer demonstrably run real large-model inference through both llama.cpp and FreeToken while correctly and observably managing an adaptive VRAM → RAM → NVMe hierarchy, and is there reproducible evidence that the mechanisms actually work?"

TierInfer is complete only when the answer is supported by the repository, tests, runtime evidence, telemetry, benchmarks, and durable state.

# SCOPE Addendum — TierInfer Implementation Quality Gate and 480B MoE System Validation

> Recorded verbatim on 2026-09-18. Extends `SCOPE.md` / `docs/SCOPE-ORIGINAL.md`;
> it does not replace, weaken, defer or reinterpret any existing goal.
> Progress against it is recorded in `docs/CHECKPOINTS.md`; the audit it
> requires is `docs/AUDIT-2026-09-18.md`.

## Status

This addendum extends the existing TierInfer scope.

It does not replace, weaken, defer, or reinterpret any existing TierInfer goal or completion criterion.

The purpose of this addendum is to:

1. perform an implementation-quality audit of the TierInfer repository as it exists now;
2. identify and eliminate incomplete, simulated, stubbed, disconnected, or misleading mechanisms before further optimization work;
3. establish a trustworthy baseline of what TierInfer actually does today;
4. validate TierInfer against a substantially larger sparse MoE model that cannot simply be treated as a conventional VRAM-resident model;
5. use that model to exercise the VRAM → RAM → NVMe architecture under realistic pressure;
6. collect evidence that can guide subsequent TierInfer implementation and optimization work.

The newly downloaded validation model is:

```text
Qwen3-Coder-480B-A35B-Instruct
Quantization: Q4_K_M
Format: GGUF
Architecture: sparse MoE
Total downloaded GGUF size: approximately 290 GB decimal / 270 GiB
Shards: 6
```

Location:

```text
/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/
```

(Measured 2026-09-18: the six shards sit one level down, in the `Q4_K_M/`
subdirectory of that path.)

Files:

```text
Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf
Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00002-of-00006.gguf
Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00003-of-00006.gguf
Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00004-of-00006.gguf
Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00005-of-00006.gguf
Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00006-of-00006.gguf
```

Observed byte sizes:

```text
49973553312
49768927168
48862488352
48952192064
49721298176
42780367136
```

The model is stored on the OWC Express 4M2 RAID0 mounted at:

```text
/data/ai-data
```

This storage system is the intended primary Tier 2 storage device for TierInfer experiments.

---

# 1. Mandatory First Action — TierInfer Quality Gate

Before attempting to optimize TierInfer for the 480B model, perform a thorough implementation-quality review of the current TierInfer repository.

Do not assume that a mechanism described in SCOPE, documentation, architecture notes, interfaces, tests, class names, configuration, or comments is actually implemented.

Inspect the code itself and establish what exists in executable production paths.

The review must explicitly look for:

* TODOs;
* FIXMEs;
* stubs;
* placeholders;
* pass-through implementations;
* methods returning constant or synthetic values;
* fake telemetry;
* mock implementations accidentally reachable from production;
* unimplemented interfaces;
* dead code;
* unreachable code;
* unused abstractions;
* configuration options that are parsed but not honored;
* command-line options that do not affect execution;
* functions that exist but are never called;
* incomplete error handling;
* silently ignored errors;
* fallback behavior that hides implementation failures;
* tests that validate mocks rather than real behavior;
* tests that pass without exercising the intended mechanism;
* hard-coded assumptions;
* temporary development shortcuts;
* duplicated implementations that may have diverged;
* concurrency hazards;
* resource leaks;
* file descriptor leaks;
* mmap lifecycle problems;
* cache accounting errors;
* invalid memory-budget assumptions;
* incorrect byte/range calculations;
* incorrect GGUF tensor offsets;
* incorrect shard handling;
* race conditions in asynchronous I/O;
* cache eviction races;
* incorrect synchronization between prefetch and demand reads;
* telemetry that reports intended state rather than observed state;
* code paths that silently fall back to ordinary OS paging while claiming TierInfer-managed I/O.

Search both mechanically and semantically.

A text search for TODO/FIXME/stub/pass is necessary but not sufficient.

Trace the important runtime paths from their external entry points to the actual storage and inference operations.

---

# 2. Architecture-to-Code Reconciliation

Reconcile the current implementation against the effective TierInfer scope.

At minimum, inspect the implementation status of:

```text
Model Layout Inspector
Tier Manager
RAM Expert Cache
VRAM Working-Set Cache
Expert Activity Tracker
Expert Predictor
Prerouter support
Async Prefetch Engine
Storage Backend
Telemetry / Benchmarking
Automatic Configuration
llama.cpp integration
FreeToken integration
FlowRunner integration
Failure handling / graceful degradation
```

For every mechanism, classify its actual state based on evidence from the repository.

Allowed classifications are:

```text
IMPLEMENTED
PARTIALLY_IMPLEMENTED
STUBBED
DISCONNECTED
NOT_IMPLEMENTED
```

Do not classify something as IMPLEMENTED merely because an interface, test, configuration field, documentation section, or source file exists.

IMPLEMENTED means that the mechanism is reachable through the intended production execution path and performs the behavior required by the effective TierInfer scope.

For every PARTIALLY_IMPLEMENTED, STUBBED, DISCONNECTED, or NOT_IMPLEMENTED item, record:

* what is missing;
* where it is missing;
* why it matters;
* what execution path is affected;
* whether it prevents meaningful 480B validation;
* the corrective action required.

Record these findings durably through the existing scope/checkpoint mechanism.

---

# 3. No False Completion

Existing completed goals must be checked against the actual repository state.

If implementation evidence contradicts a previously completed goal, do not preserve the completed status merely because it was previously recorded as complete.

Reconcile durable state with repository reality.

A goal is complete only when its completion criteria are demonstrably satisfied by the current implementation.

Documentation must never be treated as stronger evidence than executable code and observed runtime behavior.

Tests must not be treated as sufficient evidence when they only exercise mocks, synthetic data, or isolated helper functions while the corresponding production path remains disconnected.

---

# 4. Repair Before Performance Optimization

Fix material correctness and completeness problems discovered by the quality audit before interpreting performance results.

Prioritize defects in this order:

```text
correctness
    ↓
real production-path integration
    ↓
observability
    ↓
failure safety
    ↓
performance
```

Do not optimize a mechanism that is still stubbed, simulated, disconnected, or incorrectly measured.

Do not perform broad unrelated refactoring.

Changes should remain scoped to making the existing TierInfer architecture real, correct, measurable, and suitable for the large-model validation described below.

---

# 5. Preserve a Pre-Optimization Baseline

Before making performance-oriented changes specifically for Qwen3-Coder-480B-A35B, establish and record the behavior of the current implementation after correctness repairs.

Capture:

* commit/hash;
* TierInfer configuration;
* llama.cpp build/version;
* kernel;
* NVIDIA driver;
* CUDA version;
* available VRAM;
* available RAM;
* model path;
* model size;
* GGUF shard count;
* storage mount;
* storage filesystem;
* RAID topology where observable;
* relevant memory limits;
* relevant cache settings;
* relevant prefetch settings;
* relevant tier thresholds.

This becomes the reproducible TierInfer 480B baseline.

---

# 6. Validate the Downloaded GGUF Before TierInfer Testing

Verify the downloaded model before using it as evidence about TierInfer.

At minimum:

1. verify that all six shards exist;
2. verify their sizes;
3. verify that the first shard identifies the expected model architecture and quantization;
4. verify that the selected llama.cpp build recognizes the model;
5. verify that llama.cpp discovers the complete sharded model;
6. verify that model metadata can be loaded successfully;
7. verify that a minimal inference can be initialized.

The currently known llama.cpp candidate is:

```text
/home/svend/llama.cpp-qwen38/build/bin/llama-cli
```

Previously observed build:

```text
b10482-8b8640097
```

Do not assume that this build fully supports the downloaded model.

Test it.

If the build is incompatible, diagnose the incompatibility before changing llama.cpp.

Do not upgrade or replace a working runtime merely because a newer version exists.

Make the smallest necessary change.

---

# 7. Initial Runtime Compatibility Test

The first compatibility test should be deliberately conservative.

Start from the first GGUF shard:

```text
/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf
```

Initially use:

```text
-ngl 0
small context
minimal generation
default mmap behavior
```

Do not use `--no-mmap`.

The purpose is not performance.

The purpose is to establish that the model can be mapped, all shards can be resolved, the architecture is supported, and inference initialization succeeds.

Record failures precisely rather than masking them with retries or unrelated configuration changes.

---

# 8. Establish the Native llama.cpp Baseline

Before attributing any behavior to TierInfer, establish a native llama.cpp baseline using the same model and storage.

The baseline must intentionally allow ordinary Linux mmap/page-fault behavior.

Measure at least:

```text
model initialization time
prompt processing rate
generation tokens/second
RAM usage
VRAM usage
NVMe read throughput
NVMe read IOPS
average read request size
read latency where available
device utilization
CPU utilization
```

Observe both:

```text
md0
physical 4M2 member devices
```

This distinction matters because previous experiments showed that the md RAID layer can merge/coalesce many logical small reads into substantially larger physical requests.

Do not equate logical request size with physical device request size.

---

# 9. Existing Storage Evidence

Previous experiments have already established that the OWC Express 4M2 RAID0 is viable as the TierInfer Tier 2 device.

Relevant observed storage behavior includes approximately:

```text
1 MiB random reads:
~2038 IOPS
~2039 MiB/s
~2138 MB/s

4 KiB random reads:
~111k IOPS
~435 MiB/s
~456 MB/s
```

A previous constrained GLM-4.5-Air IQ4_XS inference experiment on this storage also demonstrated real active model reads during token generation.

Observed model-load behavior was approximately:

```text
~2.04 GB/s
~512 KiB reads at md0
near full device utilization
```

Under memory pressure during inference, observed md0 activity reached approximately:

```text
~140k–160k logical reads/s
~1.4–1.5 GB/s
~9–10 KiB logical average request size
```

The physical RAID members simultaneously showed substantially fewer reads with much larger average physical request sizes because of request merging.

The constrained GLM run produced approximately:

```text
Prompt:      1.8 tokens/s
Generation:  9.9 tokens/s
```

These results are evidence that the storage path is capable enough to justify further TierInfer work.

They are not proof that TierInfer itself is already optimizing these accesses.

The 480B validation must distinguish:

```text
Linux behavior
md RAID behavior
llama.cpp behavior
TierInfer behavior
```

as far as practical.

---

# 10. Large-Model Memory Pressure Is Intentional

The Qwen3-Coder model was selected specifically because it is much larger than the previous approximately 57 GB GLM test model.

The system has approximately:

```text
RTX 5090: 32 GB VRAM
System RAM: 192 GB
Model GGUF: ~270 GiB
```

The exact usable capacities and runtime allocations must be measured rather than inferred from these nominal numbers.

The test should create a genuine hierarchy in which some model data cannot remain resident in VRAM and RAM simultaneously.

Conceptually:

```text
              Qwen3-Coder-480B-A35B
                     ~270 GiB
                         │
                         ▼
                ┌────────────────┐
                │    TierInfer   │
                └───────┬────────┘
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Tier 0      Tier 1      Tier 2
           VRAM         RAM         NVMe
        RTX 5090      system      OWC 4M2
```

Do not impose an artificial 52 GB RAM limit during the first large-model baseline unless required for a specific controlled experiment.

The value of this model is that its size creates meaningful memory pressure naturally.

---

# 11. Determine the Practical GPU Offload Point

After CPU/mmap compatibility is established, determine a safe practical GPU offload configuration.

Do not immediately use an arbitrarily large `-ngl`.

Increase GPU offload systematically while observing:

```text
VRAM allocation
CUDA allocation failures
host RAM pressure
model-load behavior
generation performance
NVMe traffic
```

Preserve enough VRAM for runtime overhead and KV cache.

Record the selected configuration and the reason for selecting it.

The objective is not simply to maximize `-ngl`.

The objective is to establish a stable baseline suitable for comparing native behavior with TierInfer-managed behavior.

---

# 12. Prove Real TierInfer Data Movement

For the large-model TierInfer run, telemetry must demonstrate actual movement through the intended hierarchy.

Where implemented, collect evidence for:

```text
NVMe → RAM
RAM → VRAM
VRAM residency
RAM residency
cache hit
cache miss
prefetch issued
prefetch useful
prefetch late
prefetch unused
eviction
demand fallback
bytes transferred
transfer latency
expert identity
expert activation
```

Do not report a cache hit unless the requested data was actually served from that cache.

Do not report a prefetch hit merely because a prefetch request was issued.

A useful prefetch means the data arrived before demand required it and was subsequently consumed.

Telemetry must reflect observed execution rather than intended policy.

---

# 13. Validate Expert-Aware Behavior

Qwen3-Coder-480B-A35B is being introduced specifically to exercise sparse-MoE behavior.

Determine what expert-level information TierInfer can reliably obtain from the GGUF/runtime integration.

Validate:

```text
expert identification
expert tensor/range mapping
expert activation observation
expert frequency tracking
hot/cold classification
expert residency decisions
expert prefetch decisions
expert eviction decisions
```

If current llama.cpp integration does not expose sufficient expert-routing information, document the exact missing integration point and implement the smallest appropriate instrumentation/interface required by the effective TierInfer scope.

Do not fabricate expert identities from file-access patterns when the mapping is not known.

---

# 14. Evaluate the Expert Predictor and Prerouter

If the current TierInfer implementation contains an Expert Predictor or prerouter mechanism, validate that it is actually connected to inference.

Measure at minimum:

```text
prediction count
correct predictions
incorrect predictions
prediction precision
prediction recall where meaningful
prefetches caused by prediction
useful predicted prefetches
late predicted prefetches
wasted predicted prefetches
```

If the predictor is currently heuristic, record that fact.

If it is trainable, verify that the training and inference paths are real and connected.

If the mechanism is stubbed or disconnected, repair it according to the existing TierInfer scope before claiming predictor results.

---

# 15. Measure Coalescing Explicitly

Previous 4M2 experiments revealed that Linux/md already performs meaningful request merging.

TierInfer therefore cannot claim success merely because physical requests are larger than logical page faults.

Measure TierInfer's contribution separately where possible.

Track:

```text
logical requested ranges
TierInfer coalesced ranges
TierInfer issued I/O size
md0 observed request size
physical-device request size
```

The important comparison is:

```text
native mmap/page-fault baseline
             versus
TierInfer-managed access
```

not:

```text
4 KiB theoretical paging
             versus
physical RAID request size
```

TierInfer should demonstrate additional useful locality, batching, prediction, cache reuse, or overlap beyond what the operating system and RAID layer already provide.

---

# 16. Compare Native and TierInfer Runs

Use the same model, prompt class, context configuration, GPU configuration, storage, and system state wherever practical.

Compare:

```text
Native llama.cpp
        │
        ▼
Linux mmap/page faults
        │
        ▼
OWC 4M2

versus

llama.cpp + TierInfer
        │
        ▼
TierInfer policy/cache/prefetch
        │
        ▼
VRAM ↔ RAM ↔ OWC 4M2
```

Measure differences in:

```text
time to first token
prompt tokens/s
generation tokens/s
NVMe bytes read/token
NVMe IOPS
average I/O size
RAM cache hit rate
VRAM working-set hit rate
prefetch usefulness
stall time attributable to storage
CPU overhead
stability
```

Do not claim a performance improvement unless repeated measurements support it.

Single-run token-generation rates are not sufficient because sparse MoE routing and generated token sequences can alter the workload.

---

# 17. Repeated Measurements

Performance conclusions must use repeated runs.

Use reproducible prompts and configurations.

Separate:

```text
cold-cache runs
warm-cache runs
```

when relevant.

Record enough information to determine whether data may have remained in Linux page cache or TierInfer caches.

Where practical, report:

```text
median
minimum
maximum
run count
```

for key performance metrics.

Do not silently select the fastest run.

---

# 18. Graceful Degradation

The large model is also a failure-safety test.

Validate behavior when:

```text
VRAM is exhausted
RAM pressure increases
prefetch falls behind
NVMe latency spikes
a requested expert is not cached
prediction is wrong
cache space is exhausted
an asynchronous read fails
a model range cannot be resolved
```

TierInfer must not silently corrupt inference.

Failure should either:

```text
fall back to a correct slower path
```

or:

```text
fail explicitly with actionable diagnostics
```

according to the effective TierInfer scope.

---

# 19. Protect the Host System

Do not destabilize the workstation merely to obtain a benchmark.

The machine is a development workstation, not a disposable benchmark node.

Avoid uncontrolled OOM conditions.

Do not globally disable swap merely to force TierInfer behavior.

Do not globally alter kernel memory policy without a demonstrated requirement.

Prefer process-scoped or cgroup-scoped controls when controlled memory-pressure experiments are necessary.

Do not delete existing models or unrelated data to make room without explicit approval.

---

# 20. No Unnecessary Re-Download

The 480B Q4_K_M model is already downloaded successfully.

Do not download another copy unless integrity validation demonstrates that the existing copy is defective.

Do not automatically replace it with another quantization.

Do not download DeepSeek-V3.1 or another several-hundred-gigabyte model as part of this addendum.

The current model is the designated large-model TierInfer validation target.

---

# 21. Evidence-Based Checkpoints

Create durable checkpoints at meaningful boundaries, including at least:

```text
quality audit complete
correctness repairs complete
480B GGUF validated
native llama.cpp baseline complete
GPU offload baseline selected
TierInfer large-model run operational
expert telemetry validated
prefetch/cache telemetry validated
native-vs-TierInfer comparison complete
final reconciliation complete
```

Each checkpoint must contain:

* repository revision;
* what was tested;
* exact command/configuration where relevant;
* result;
* measured evidence;
* unresolved problems;
* next action.

The work must be resumable from durable state without relying on conversational context.

---

# 22. Required Quality-Audit Deliverable

Before proceeding deeply into the 480B optimization work, produce a concise implementation audit containing:

```text
Component
Expected behavior
Actual implementation status
Evidence
Defects/gaps
Required correction
Blocking/non-blocking
```

The audit is not a documentation exercise.

Its purpose is to establish whether TierInfer currently contains real working mechanisms or merely architectural scaffolding.

Any significant discrepancy between SCOPE and implementation becomes actionable work under the existing scope.

---

# 23. Required 480B Validation Deliverable

At completion of this addendum, produce an evidence-based validation summary containing at least:

```text
Model
Quantization
Model size
llama.cpp version
TierInfer revision
VRAM configuration
RAM behavior
NVMe behavior
native generation performance
TierInfer generation performance
native I/O characteristics
TierInfer I/O characteristics
RAM cache effectiveness
VRAM working-set effectiveness
prefetch effectiveness
expert predictor effectiveness
observed bottlenecks
correctness issues discovered
correctness issues repaired
remaining scope gaps
```

Do not turn this into a marketing-style success report.

A negative result is valid evidence.

If TierInfer is slower than native mmap, report that.

If a cache provides no measurable benefit, report that.

If prediction is ineffective, report that.

If an architectural assumption is wrong, record the decision and adapt the implementation according to the effective scope.

---

# 24. Acceptance Criteria

This addendum is complete only when all of the following are true:

1. The current TierInfer repository has undergone a code-level quality audit.
2. Material stubs, placeholders, fake implementations, disconnected production paths, misleading telemetry, and correctness defects relevant to the effective TierInfer scope have been identified.
3. Blocking correctness/integration defects have been repaired.
4. Existing durable goal status has been reconciled against actual implementation evidence.
5. The six downloaded Qwen3-Coder-480B-A35B Q4_K_M shards have been validated.
6. The selected llama.cpp runtime can load/initialize the model, or a precisely diagnosed compatibility issue and required minimal correction has been established and resolved.
7. A reproducible native llama.cpp baseline has been measured.
8. A stable GPU-offload configuration has been established.
9. The model has been exercised under genuine VRAM/RAM/NVMe pressure.
10. TierInfer's actual data movement can be distinguished from ordinary Linux mmap/page-fault behavior sufficiently to evaluate the implementation.
11. RAM cache behavior is measured from real accesses.
12. VRAM working-set behavior is measured from real accesses.
13. Expert activity is observed from real model execution where the runtime provides or can reasonably expose it.
14. Prefetch behavior is measured rather than inferred.
15. Predictor/prerouter behavior is verified as real, connected, and measurable if present in the effective implementation scope.
16. Native llama.cpp and TierInfer behavior have been compared using repeated controlled measurements.
17. Existing Linux/md RAID request merging has been accounted for when interpreting TierInfer I/O improvements.
18. Failure behavior has been tested sufficiently to establish that TierInfer fails safely or degrades correctly.
19. Results and unresolved gaps have been recorded durably.
20. The repository, tests, durable scope state, and implementation status agree at the end of the work.

---

# 25. Execution Instruction to Fable 5

Begin with the implementation quality gate.

Do not begin by tuning the 480B model.

First determine whether the mechanisms already built in TierInfer are real, complete, connected, correctly measured, and consistent with the effective scope.

Inspect the repository rather than trusting previous completion statements.

Repair material correctness and production-path defects before performance optimization.

Then validate the downloaded Qwen3-Coder-480B-A35B Q4_K_M model conservatively.

Establish native llama.cpp behavior first.

Only after that baseline is trustworthy should TierInfer-managed VRAM/RAM/NVMe behavior be evaluated and optimized.

Use the existing durable scope/checkpoint mechanisms throughout.

Keep changes minimal, evidence-driven, testable, and within the effective TierInfer scope.

The objective is not to make TierInfer appear successful.

The objective is to determine what TierInfer actually does, correct what is incomplete, and then prove whether its explicit VRAM → RAM → NVMe architecture provides a real and measurable advantage on a large sparse MoE workload.

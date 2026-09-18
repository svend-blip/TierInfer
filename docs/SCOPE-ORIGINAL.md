# TierInfer — Adaptive VRAM/RAM/NVMe Inference for Large Local Models

> The Human's original scope statement, recorded verbatim on 2026-09-18 so
> the repository carries the document it is measured against. `SCOPE.md` is
> the working condensation with per-goal status; where the two differ in
> what is *required*, this file rules.

## 1. Project Identity

**Project name:** TierInfer
**Repository:** `https://github.com/svend-blip/TierInfer.git`
**Local project directory:** `TierInfer`

TierInfer is an adaptive inference memory and streaming layer for running large local language models across a hierarchical memory system consisting of:

```text
VRAM
  ↓
RAM
  ↓
NVMe
```

The project treats NVMe storage as an active inference resource rather than only as a location from which a model is initially loaded.

TierInfer must provide a reusable implementation that can operate with:

* llama.cpp
* FreeToken
* FlowRunner

The implementation must not be tied to one model, one quantization format, one runtime, or one specific GPU configuration.

TierInfer must support large dense models where tiered storage is useful, but its primary optimization target is sparse Mixture-of-Experts inference, where only a subset of expert weights is required for each token and therefore expert-aware caching and prefetching can substantially reduce active memory requirements.

---

# 2. Origin and Primary Inspiration

The original source that motivated this project is:

**Edge0**
`https://github.com/Edge0-AI/Edge0`

Edge0 demonstrates a streaming Mixture-of-Experts inference architecture in which expert weights can remain on SSD and are loaded on demand instead of requiring the entire quantized model to reside in active memory.

Relevant Edge0 mechanisms include:

* SSD expert offload
* memory-mapped model data
* shared expert caching
* hot-expert residency
* asynchronous or staged expert loading
* prefetching
* expert routing prediction
* a prerouter that predicts required experts ahead of their use
* overlap of storage I/O and model computation
* bounded active memory determined by the working set instead of total model size

TierInfer is not intended to reproduce Edge0 as-is.

Edge0 is the originating technical inspiration and reference architecture.

TierInfer generalizes the underlying concept into an adaptive VRAM/RAM/NVMe hierarchy suitable for Linux/CUDA local inference environments and integration with existing inference runtimes.

The project must independently define and implement its own runtime abstractions, measurements, policies, adapters, and integration boundaries.

---

# 3. Problem Statement

Local inference capacity is normally constrained by two expensive resources:

1. GPU VRAM
2. system RAM

NVMe storage is substantially cheaper per gigabyte and can provide several GB/s of sequential bandwidth together with high random-read IOPS.

Traditional local inference normally uses storage primarily during model loading:

```text
NVMe
  ↓
RAM / mmap page cache
  ↓
VRAM / CPU execution
```

Once model pages are resident in RAM, storage usually plays little or no active role in token generation.

This creates a practical limitation:

```text
model size
    >
available VRAM + practical RAM allocation
    =
model cannot be efficiently served
```

For sparse Mixture-of-Experts models this is unnecessarily restrictive.

A sparse MoE model may contain hundreds of experts while only a small subset is executed for each token.

The full parameter set therefore does not have to remain in expensive memory if TierInfer can determine:

* which experts must remain in VRAM,
* which experts should remain cached in RAM,
* which experts may remain on NVMe,
* which experts are likely to be required next,
* when expert data should be prefetched,
* when expert data should be promoted or demoted between tiers.

TierInfer must turn this observation into a practical local inference system.

---

# 4. Initial Investigation and Experimental Evidence

Before creating TierInfer, direct experiments were performed on a Linux AI workstation using:

* NVIDIA RTX 5090 32 GB
* approximately 192 GB system RAM
* Samsung 990 PRO NVMe SSD
* llama.cpp
* GLM-4.5-Air IQ4_XS GGUF
* model size approximately 57 GB
* partial CUDA offload

The investigation established several important facts.

## 4.1 Large-model partial GPU offload works

The approximately 57 GB model could not fit entirely in 32 GB VRAM.

Attempting aggressive GPU offload resulted in CUDA out-of-memory errors.

A stable configuration using:

```text
-ngl 25
```

successfully split inference between GPU and CPU-side memory.

With unrestricted system memory, measured generation performance was approximately:

```text
7.7 tokens/second
```

This established the normal VRAM + RAM baseline.

---

## 4.2 NVMe model loading bandwidth was measured

After clearing Linux page cache and starting the model cold, the Samsung 990 PRO delivered approximately:

```text
2.4–3.1 GB/s
```

during model loading.

This verified that the storage path can supply model data at multi-gigabyte-per-second throughput.

Once the model pages became resident, normal token generation required very little SSD activity.

This confirmed the conventional behavior:

```text
NVMe
  ↓
initial page population
  ↓
RAM page cache
  ↓
inference
```

---

## 4.3 RAM pressure caused active NVMe inference paging

The same workload was then executed inside a constrained systemd memory scope:

```text
MemoryHigh = 48 GB
MemoryMax  = 52 GB
```

The system could no longer retain the entire CPU-side model working set in RAM.

During inference, sustained NVMe activity appeared.

Observed behavior included approximately:

```text
500–830 MB/s sustained reads
```

with extremely high read operation counts, at times approaching:

```text
~200,000 reads/second
```

The individual operations were predominantly approximately:

```text
4 KB
```

page-sized reads.

Generation continued successfully at approximately:

```text
5.5 tokens/second
```

compared with the unrestricted-memory baseline of approximately:

```text
7.7 tokens/second
```

This experiment proved the central technical premise of TierInfer:

> NVMe can operate as an active backing memory tier during local LLM inference, rather than merely as model-load storage.

It also exposed the central inefficiency that TierInfer must eliminate.

Linux generic mmap/page-fault paging can make NVMe-backed inference work, but it produces large numbers of small, demand-driven reads.

The storage system is reacting after a required page is missing.

TierInfer must replace reactive generic paging with model-aware, expert-aware and predictive movement of weights.

---

# 5. Core Objective

TierInfer must implement an adaptive three-tier inference memory hierarchy:

```text
Tier 0 — VRAM
Tier 1 — RAM
Tier 2 — NVMe
```

The system must continuously manage model data across these tiers according to actual inference demand.

The resulting system must make it possible to run models whose total weight set exceeds practical VRAM and RAM capacity while preserving useful interactive inference performance.

For sparse MoE models, TierInfer must operate at expert granularity wherever the underlying runtime and model representation permit it.

The system must minimize synchronous storage stalls by predicting, prefetching, caching and promoting model weights before they are required.

TierInfer must achieve the complete functional chain:

```text
model stored on NVMe
        ↓
model metadata indexed
        ↓
active working set identified
        ↓
hot experts cached in RAM
        ↓
hottest experts promoted to VRAM where appropriate
        ↓
next experts predicted or inferred
        ↓
asynchronous prefetch initiated
        ↓
storage I/O overlaps model computation
        ↓
required expert is resident before execution
        ↓
unused experts are demoted or evicted
        ↓
policy continuously adapts from runtime behavior
```

---

# 6. Architectural Principles

## 6.1 Runtime independence

TierInfer must not become a fork-specific implementation embedded permanently inside one inference runtime.

The project must define a reusable core with runtime adapters.

Conceptually:

```text
                    TierInfer Core
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
 llama.cpp adapter   FreeToken adapter   FlowRunner adapter
```

Runtime-specific implementations may use native extension points or patches where required, but memory policy, metrics, cache policy and tier-management logic must remain reusable wherever technically possible.

---

## 6.2 Explicit memory tiers

The system must model VRAM, RAM and NVMe as explicit resources.

Each tier must expose at least:

* capacity
* available capacity
* current occupancy
* object residency
* promotion cost
* demotion cost
* transfer bandwidth
* transfer latency
* current transfer pressure
* cache hit/miss information

The system must not treat RAM and storage as an opaque operating-system-managed mmap pool when a more explicit mechanism can be used.

---

## 6.3 Active working set instead of total parameter count

The design must distinguish between:

```text
total model weights
```

and:

```text
weights required by the current inference working set
```

For MoE models this distinction must extend to individual experts or expert groups.

TierInfer must optimize active memory according to the working set rather than total model size.

---

## 6.4 Model-aware I/O

TierInfer must avoid relying solely on random operating-system page faults.

The storage layer must understand logical model objects such as:

* layers
* tensors
* experts
* expert groups
* shared experts
* attention weights
* routing weights

Reads should be coalesced into larger efficient operations wherever possible.

The system must support asynchronous read scheduling and explicit prefetch.

---

## 6.5 Adaptive policy

Static rules alone are insufficient.

TierInfer must collect runtime data and adapt placement decisions according to observed behavior.

Relevant signals include:

* expert activation frequency
* recent expert use
* repeated routing patterns
* cache hits
* cache misses
* NVMe read latency
* NVMe bandwidth
* RAM pressure
* VRAM pressure
* inference phase
* token generation rate
* prefetch accuracy
* eviction rate
* model topology

---

# 7. Required Subsystems

TierInfer must contain the following major subsystems.

---

## 7.1 Model Layout Inspector

TierInfer must inspect supported model formats and determine where relevant weights are stored.

For GGUF and other supported layouts, the inspector must produce an index describing:

```text
model
 ├── layers
 │    ├── attention tensors
 │    ├── shared tensors
 │    └── MoE tensors
 │          ├── expert 0
 │          ├── expert 1
 │          ├── ...
 │          └── expert N
```

The index must include byte offsets and sizes wherever direct range loading is possible.

This allows TierInfer to request logical objects instead of triggering arbitrary virtual-memory pages.

---

## 7.2 Tier Manager

The Tier Manager is responsible for weight residency.

It must maintain authoritative state such as:

```text
object X:
    NVMe = present
    RAM  = cached
    VRAM = resident
```

It must support:

* promotion NVMe → RAM
* promotion RAM → VRAM
* direct transfer paths where supported
* demotion VRAM → RAM
* eviction from RAM
* persistence on NVMe
* residency pinning
* capacity reservation
* asynchronous transfer tracking

---

## 7.3 RAM Expert Cache

TierInfer must implement a bounded RAM cache.

For MoE workloads, it must support expert-level caching.

The cache policy must use runtime evidence rather than simple file-page recency alone.

At minimum the cache policy must account for:

* recency
* frequency
* expert activation probability
* transfer cost
* expert size
* prediction confidence
* pinned or shared weights

An LRU-compatible strategy may be used as a foundation, but policy must support additional adaptive signals.

---

## 7.4 VRAM Working-Set Cache

TierInfer must manage a configurable GPU-resident working set where runtime integration permits explicit control.

VRAM must be reserved for the highest-value objects.

Examples include:

* permanently required dense tensors
* shared experts
* frequently activated routed experts
* currently executing experts
* predicted near-term experts

TierInfer must prevent expert caching from exhausting VRAM required for:

* KV cache
* activations
* CUDA graphs or equivalent runtime allocations
* temporary compute buffers

VRAM allocation policy must therefore operate within a configurable inference memory budget rather than simply consuming all available GPU memory.

---

## 7.5 Expert Activity Tracker

The system must record expert-routing behavior.

Per expert, relevant statistics include:

```text
activation count
recent activation history
moving activation frequency
last-used token
reuse distance
cache residency
prefetch success
prefetch waste
load latency
```

This activity history must feed the cache and prediction systems.

---

## 7.6 Expert Predictor

TierInfer must support predictive expert loading.

The initial predictor must be able to use deterministic runtime information and recent routing history.

TierInfer must also provide a trainable prerouter mechanism for models where predictive heads can materially improve routing prediction.

The prerouter concept is based on the original Edge0 observation that waiting for normal MoE routing before loading an expert creates an unavoidable storage stall.

TierInfer must therefore be capable of predicting likely expert requirements ahead of execution.

Conceptually:

```text
token N executing
       │
       ├── normal computation
       │
       └── predict experts for token N+1
                     │
                     ▼
              async NVMe read
                     │
                     ▼
                 RAM cache
                     │
                     ▼
            optional VRAM promote

token N+1 arrives
       │
       ▼
expert already resident
```

Prediction accuracy and performance impact must be measurable.

---

## 7.7 Async Prefetch Engine

TierInfer must provide an asynchronous prefetch engine.

The engine must:

* accept predicted objects
* prioritize reads
* coalesce adjacent reads
* avoid duplicate requests
* cancel or deprioritize obsolete requests where possible
* track in-flight transfers
* expose completion state
* overlap I/O with inference compute

The prefetch system must distinguish between:

```text
required-now
high-confidence-next
probable-next
background-hot
```

priority classes.

---

## 7.8 Storage Backend

NVMe must be treated as a configurable backend.

TierInfer must collect real runtime characteristics instead of assuming fixed drive performance.

The backend must measure or expose:

* sequential throughput
* random-read throughput
* IOPS
* average latency
* queue depth
* effective read size
* device utilization

TierInfer should exploit large and coalesced reads when model layout permits them.

The project must specifically avoid the pathological pattern observed in the baseline constrained-memory test:

```text
hundreds of thousands of reactive 4 KB reads
```

when a predictable expert-level read can replace them.

---

## 7.9 Telemetry and Benchmarking

Performance instrumentation is part of the product, not an optional debugging feature.

TierInfer must expose measurements for:

```text
prompt tokens/second
generation tokens/second

VRAM occupancy
RAM cache occupancy
NVMe occupancy

VRAM cache hit rate
RAM cache hit rate
NVMe miss/read rate

prefetch requests
prefetch hits
prefetch misses
prefetch waste

bytes NVMe → RAM
bytes RAM → VRAM

read size distribution
I/O latency
I/O bandwidth
queue depth

expert activation distribution
expert residency distribution
```

Metrics must be available in machine-readable form for integration with other tools.

---

# 8. Runtime Integration

## 8.1 llama.cpp

TierInfer must work with llama.cpp.

The integration must provide a path from standard llama.cpp inference to TierInfer-controlled memory behavior.

The llama.cpp adapter must support:

* GGUF metadata inspection
* tensor and expert location mapping
* partial GPU offload
* TierInfer-controlled expert residency where technically possible
* asynchronous weight loading
* benchmark comparison against normal mmap behavior

The implementation must preserve a clean baseline mode in which llama.cpp operates normally without TierInfer.

This allows direct A/B comparison.

Example:

```text
llama.cpp normal mmap
        versus
llama.cpp + TierInfer
```

The first major llama.cpp test workload must reproduce the already-established GLM-4.5-Air baseline so that improvements can be measured against:

```text
unrestricted RAM:
~7.7 tok/s

forced Linux NVMe paging:
~5.5 tok/s
```

---

## 8.2 FreeToken

TierInfer must work with FreeToken.

FreeToken already provides mechanisms designed for models that exceed local GPU capacity.

TierInfer must complement these capabilities by providing explicit storage-aware model placement and expert streaming rather than treating system memory as the final offload tier.

The target hierarchy is:

```text
GPU execution / VRAM
        ↓
FreeToken CPU/RAM offload
        ↓
TierInfer RAM expert cache
        ↓
TierInfer NVMe backing store
```

The integration must expose enough control to avoid duplicate caching and conflicting memory-management policies.

TierInfer must be able to receive or derive relevant model-execution information from FreeToken and use it to drive:

* expert cache decisions
* expert prefetch
* storage scheduling
* telemetry

---

## 8.3 FlowRunner

TierInfer must integrate with FlowRunner as an inference capability.

FlowRunner must be able to select a TierInfer-enabled runtime for a model or flow.

TierInfer configuration must therefore be expressible declaratively.

A conceptual configuration may include:

```yaml
inference:
  runtime: llama.cpp
  tierinfer:
    enabled: true

    vram:
      budget: auto

    ram:
      cache: auto

    nvme:
      path: /path/to/model

    expert_cache:
      policy: adaptive

    prefetch:
      enabled: true

    predictor:
      enabled: true
```

Exact schema may differ, but the capability must be representable in FlowRunner configuration without hard-coded runtime-specific logic in the flow itself.

FlowRunner must be able to consume TierInfer telemetry so that execution results can report:

* runtime used
* model
* effective tier allocations
* tokens/second
* NVMe activity
* cache efficiency
* predictor efficiency

---

# 9. Execution Modes

TierInfer must provide multiple execution modes for controlled testing and production use.

## Baseline mode

```text
TierInfer disabled
```

Used to measure the native runtime.

---

## Observability mode

TierInfer observes:

* routing
* memory
* storage
* residency

but does not alter placement.

This establishes trustworthy workload traces.

---

## Cache mode

TierInfer controls RAM cache residency without predictive loading.

---

## Prefetch mode

TierInfer controls RAM residency and asynchronously prefetches likely required weights.

---

## Full adaptive mode

TierInfer controls:

```text
VRAM working set
RAM cache
NVMe storage
expert prediction
prefetch
promotion
demotion
eviction
```

as one adaptive system.

All modes belong to the completed TierInfer solution and must remain available for diagnostics and benchmarking.

---

# 10. Required Deliverables and Subgoals

## Goal 1 — Repository and project foundation

Establish the TierInfer repository and project structure in the existing `TierInfer` directory and repository:

`https://github.com/svend-blip/TierInfer.git`

Deliver:

* project metadata
* README
* SCOPE.md
* source structure
* tests
* benchmark structure
* reproducible development environment
* architecture documentation

---

## Goal 2 — Reproducible baseline benchmark

Automate the experiment already performed manually.

The benchmark must reproduce and record:

```text
model
runtime
GPU offload
RAM conditions
NVMe conditions
prompt throughput
decode throughput
NVMe bandwidth
I/O size
IOPS
VRAM use
RAM use
```

The benchmark must distinguish:

```text
warm cache
cold cache
RAM-constrained
TierInfer-enabled
```

runs.

---

## Goal 3 — Model-layout indexing

Implement model inspection and byte-range indexing.

For supported MoE models, TierInfer must identify expert weights individually.

The index must permit logical reads such as:

```text
load expert 37 from layer 18
```

rather than generic page-fault access.

---

## Goal 4 — Explicit NVMe-to-RAM expert streaming

Implement explicit asynchronous storage reads.

Prove that a required expert can be read from NVMe into a controlled RAM cache independently of Linux demand paging.

Collect and expose transfer timing.

---

## Goal 5 — RAM expert cache

Implement bounded expert-aware RAM caching.

Demonstrate:

* cache insertion
* cache hit
* cache miss
* eviction
* pinning
* adaptive hot-expert retention

---

## Goal 6 — llama.cpp integration

Integrate TierInfer with llama.cpp.

The integration must execute real model inference while TierInfer controls or assists relevant weight residency and streaming operations.

The result must be benchmarkable directly against native llama.cpp mmap behavior.

---

## Goal 7 — Asynchronous expert prefetch

Implement read-ahead of experts before their execution point.

I/O must overlap model computation.

Measure:

```text
synchronous wait eliminated
prefetch accuracy
prefetch lead time
wasted prefetch bandwidth
decode throughput
```

---

## Goal 8 — Expert prediction and prerouter

Implement predictive routing support.

The predictor must provide a probability-ranked or selected set of likely upcoming experts.

The complete solution must support a trainable prerouter approach where that provides superior prediction.

Prediction must be evaluated against actual routing.

Metrics must include:

```text
top-k prediction accuracy
expert coverage
false-positive loads
missed experts
tokens/second impact
```

---

## Goal 9 — VRAM expert working set

Implement explicit hot-weight residency in GPU memory where supported.

TierInfer must automatically maintain a safe VRAM budget.

The system must coordinate:

```text
permanent GPU weights
expert cache
KV cache
runtime workspace
temporary buffers
```

without causing unstable OOM behavior.

---

## Goal 10 — Adaptive tier policy

Combine runtime signals into a unified adaptive policy.

The policy must dynamically decide:

```text
keep in VRAM
keep in RAM
prefetch to RAM
promote to VRAM
demote to RAM
evict from RAM
leave on NVMe
```

The policy must adapt during inference instead of relying exclusively on startup configuration.

---

## Goal 11 — FreeToken integration

Provide a FreeToken adapter.

The adapter must make NVMe a usable backing tier beneath FreeToken's RAM/VRAM memory strategy.

TierInfer must expose or consume the information necessary to avoid redundant transfers and duplicated caches.

Real inference benchmarks must verify the integration.

---

## Goal 12 — FlowRunner integration

Expose TierInfer as a reusable FlowRunner inference capability.

Flows must be able to request TierInfer without implementing memory-management logic themselves.

Runtime configuration and telemetry must pass cleanly through FlowRunner.

---

## Goal 13 — Unified telemetry

Implement a common telemetry interface across supported runtimes.

A TierInfer inference run must be capable of producing a report similar to:

```text
TierInfer Run

Model:
GLM-4.5-Air IQ4_XS

Runtime:
llama.cpp

VRAM:
29.4 GiB / 32 GiB

RAM expert cache:
31.7 GiB / 48 GiB

NVMe:
Samsung 990 PRO

Decode:
X.X tok/s

RAM hit:
XX %

VRAM hit:
XX %

NVMe reads:
XXX MB/s

Average expert read:
XX MB

Prefetch accuracy:
XX %

Synchronous expert stalls:
XX ms/token
```

The telemetry schema must remain runtime-independent.

---

## Goal 14 — Automatic configuration

TierInfer must determine safe defaults from the host.

The system must inspect:

* GPU type
* available VRAM
* system RAM
* NVMe characteristics
* model size
* model topology
* quantization
* runtime
* requested context size

From these values it must derive initial:

```text
VRAM budget
RAM cache budget
NVMe strategy
prefetch concurrency
cache policy
expert residency
```

Manual overrides must remain possible.

---

## Goal 15 — Failure safety and graceful degradation

TierInfer must never depend on perfect prediction.

A missing prediction must fall back safely to an exact load.

If predictive loading fails:

```text
prediction miss
      ↓
exact expert requested
      ↓
load expert
      ↓
continue inference
```

The implementation must favor reduced performance over incorrect model execution.

TierInfer must also safely handle:

* missing model metadata
* unsupported tensor layouts
* storage errors
* insufficient VRAM
* insufficient RAM
* prefetch queue saturation
* predictor failure
* runtime adapter failure

---

# 11. Performance Objective

The initial reference experiment establishes these practical baselines:

```text
GLM-4.5-Air IQ4_XS
RTX 5090 32 GB
llama.cpp
-ngl 25
```

Normal RAM-backed inference:

```text
~7.7 generation tok/s
```

Generic Linux NVMe paging under memory pressure:

```text
~5.5 generation tok/s
```

TierInfer must demonstrate measurable improvement over generic reactive Linux paging under equivalent constrained-memory conditions.

The implementation must specifically seek to reduce:

* number of storage operations
* 4 KB random page-fault reads
* synchronous storage stalls
* cache churn
* unnecessary expert transfers

while increasing:

* effective read size
* useful NVMe bandwidth
* expert cache hit rate
* prefetch hit rate
* I/O/compute overlap
* decode throughput

The target is not merely to make an oversized model execute.

The target is to make NVMe-backed inference a deliberately engineered execution mode.

---

# 12. Correctness Requirement

Performance optimizations must not alter model semantics.

For identical model state and inference settings, TierInfer must preserve correct expert selection and model execution.

Prediction may determine what is prefetched.

Prediction must not silently replace the authoritative routing result unless an explicitly supported model/runtime mechanism guarantees equivalent intended semantics.

Fallback to exact routing and exact weight loading must always remain possible.

---

# 13. Compatibility Objective

TierInfer must support the complete software path:

```text
                  FlowRunner
                      │
                      ▼
                  TierInfer
                      │
          ┌───────────┴───────────┐
          │                       │
      llama.cpp               FreeToken
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
            VRAM / RAM / NVMe
```

TierInfer must remain useful independently of FlowRunner.

llama.cpp and FreeToken integrations must therefore be usable directly from their own execution environments.

FlowRunner provides orchestration and configuration above TierInfer rather than becoming a hard dependency of the TierInfer core.

---

# 14. Scope Boundaries

TierInfer is responsible for:

* inference memory tiering
* model-weight residency
* expert-aware caching
* storage streaming
* prefetch
* routing prediction
* transfer scheduling
* runtime integration
* telemetry
* adaptive memory policy

TierInfer is not a replacement for:

* llama.cpp
* FreeToken
* FlowRunner
* CUDA
* model training frameworks
* model file formats

It acts as the memory and streaming intelligence layer between model inference runtimes and heterogeneous local memory/storage resources.

---

# 15. Completion Definition

TierInfer is complete when the repository contains one coherent implementation that can:

1. inspect a supported large local model,
2. identify relevant model weights and MoE experts,
3. manage an explicit VRAM/RAM/NVMe hierarchy,
4. stream required weights from NVMe,
5. maintain bounded RAM and VRAM working sets,
6. track expert usage,
7. predict upcoming expert demand,
8. asynchronously prefetch predicted weights,
9. overlap storage transfer with inference,
10. safely recover from prediction misses,
11. dynamically promote and demote weights between tiers,
12. expose comprehensive runtime telemetry,
13. automatically derive safe host-specific memory budgets,
14. operate with llama.cpp,
15. operate with FreeToken,
16. operate as a selectable capability in FlowRunner,
17. benchmark itself against native runtime behavior,
18. demonstrate measurable improvement over generic Linux demand paging under constrained-memory inference.

The complete TierInfer system must establish NVMe as an intentional, model-aware third inference memory tier rather than an accidental operating-system swap mechanism.

**TierInfer turns VRAM, system RAM and NVMe into one adaptive inference memory hierarchy for large local models.**

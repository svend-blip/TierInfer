# APPENDIX A — Traceable Acceptance Matrix

> Recorded verbatim from the Human on 2026-09-18. Mandatory project state.
> The live status of every ID, with its evidence, is `docs/ACCEPTANCE-STATUS.md`.

## A.1 Purpose

This matrix converts the TierInfer Definition of Done into individually traceable acceptance requirements. It is mandatory project state, not optional documentation. Every Definition of Done requirement must map to one or more stable Acceptance IDs. Every Acceptance ID must be backed by objective evidence.

Fable 5 must use this matrix to: plan remaining work; connect requirements to implementation; connect implementation to tests; connect tests to runtime evidence; identify coverage gaps; prevent false completion; reconcile durable scope state; perform final acceptance.

The traceability chain is:

```text
Definition of Done → Acceptance ID → Implementation → Automated Test → Runtime Validation → Measured Evidence → Durable Checkpoint → ACCEPTED
```

No acceptance item may become ACCEPTED solely because code exists or an automated test passes.

## A.2 Acceptance Status Model

Each Acceptance ID must have exactly one current status: `NOT_STARTED`, `IN_PROGRESS`, `BLOCKED`, `IMPLEMENTED_UNVERIFIED`, `VERIFIED`, `ACCEPTED`.

- **NOT_STARTED** — No sufficient implementation exists.
- **IN_PROGRESS** — Implementation or correction is underway.
- **BLOCKED** — Completion is prevented by a documented blocker.
- **IMPLEMENTED_UNVERIFIED** — Production implementation appears complete, but required runtime evidence has not yet been obtained.
- **VERIFIED** — Required technical evidence exists, but final reconciliation/acceptance has not yet occurred.
- **ACCEPTED** — Implementation, tests, runtime evidence, documentation where required, and durable state all agree.

Only ACCEPTED satisfies final Definition of Done.

## A.3 Evidence Requirements

Each Acceptance ID must maintain traceable evidence containing, where applicable: Acceptance ID, DoD reference, Requirement, Implementation location, Relevant commit, Automated test, Runtime test, Test command, Model/workload, Configuration, Observed result, Telemetry evidence, Benchmark artifact, Durable checkpoint, Status, Blocking issue, Notes. Evidence should reference actual repository paths, commits, test names, commands, logs, benchmark artifacts, and checkpoint IDs rather than prose claims whenever possible.

## A.4 Core Architecture Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-CORE-001 | Explicit three-tier architecture exists | Production code + runtime telemetry | VRAM, RAM, and NVMe exist as distinguishable managed tiers |
| TI-CORE-002 | Tier Manager controls real decisions | Production-path trace + runtime events | Tier decisions materially affect inference |
| TI-CORE-003 | Tier state is observable | Telemetry test | Current residency and movement can be inspected |
| TI-CORE-004 | Model activity influences policy | Controlled runtime test | Different activity results in different justified residency decisions |
| TI-CORE-005 | Resource pressure influences policy | Controlled pressure test | Policy adapts without unsafe behavior |
| TI-CORE-006 | TierInfer is distinguishable from OS paging | Native/TierInfer comparison | TierInfer-managed operations can be identified independently of mmap/page faults |
| TI-CORE-007 | Shared architecture is used by runtime adapters | Code trace | llama.cpp and FreeToken do not implement unrelated duplicate tier managers |

## A.5 Model Layout Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-LAYOUT-001 | GGUF metadata inspection works | Unit + real GGUF test | Model metadata is parsed correctly |
| TI-LAYOUT-002 | Sharded GGUF models work | 480B six-shard validation | All shards are discovered and mapped correctly |
| TI-LAYOUT-003 | Tensor offsets are correct | Range-validation test | Known tensor ranges match actual GGUF locations |
| TI-LAYOUT-004 | Layer mapping works | Inspection artifact | Relevant tensors can be associated with model structure |
| TI-LAYOUT-005 | MoE structures are recognized | Qwen3-Coder runtime evidence | Relevant MoE structures are identified correctly |
| TI-LAYOUT-006 | Expert ranges are mapped where supported | Expert/range validation | Expert identity maps to correct storage data |
| TI-LAYOUT-007 | Unsupported structures fail explicitly | Negative test | No fabricated layout information is produced |

## A.6 NVMe Tier Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-NVME-001 | NVMe is an explicit Tier 2 | Production-path evidence | TierInfer directly requests model data from storage |
| TI-NVME-002 | Model-aware ranges are used | I/O trace | Reads correspond to known model ranges |
| TI-NVME-003 | Adjacent reads can be coalesced | Controlled test | Multiple useful ranges become fewer/larger requests |
| TI-NVME-004 | Batched I/O works | Integration test | Multiple required ranges can be scheduled efficiently |
| TI-NVME-005 | Async I/O works | Timing/concurrency evidence | Reads can execute without synchronously blocking the issuing path |
| TI-NVME-006 | Multiple reads can be in flight | Runtime telemetry | Bounded concurrent storage requests are observed |
| TI-NVME-007 | Storage errors are handled safely | Failure injection | Failed I/O cannot cause silent incorrect inference |
| TI-NVME-008 | TierInfer contribution exceeds simple relabeling of md/Linux behavior | Native comparison | Measured TierInfer behavior is distinguishable from native request merging/page cache |

## A.7 RAM Cache Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-RAM-001 | Real bounded RAM cache exists | Production code + memory test | Cache capacity is enforced |
| TI-RAM-002 | Cache admission works | Runtime trace | Requested model data can enter the cache |
| TI-RAM-003 | Cache lookup works | Hit test | Reused data is served from RAM |
| TI-RAM-004 | Hits/misses are real | Telemetry correlation | Counters correspond to observed accesses |
| TI-RAM-005 | Eviction works | Capacity-pressure test | Data is evicted according to policy |
| TI-RAM-006 | Accounting is correct | Unit + pressure test | Reported bytes correspond to actual residency |
| TI-RAM-007 | Concurrent access is safe | Concurrency tests | No races/corruption under concurrent reads |
| TI-RAM-008 | Expert caching works where supported | Sparse-MoE test | Reused expert data benefits from RAM residency |

## A.8 VRAM Working-Set Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-VRAM-001 | VRAM residency is observable | Runtime telemetry | TierInfer can identify relevant resident model data |
| TI-VRAM-002 | Promotion works | Runtime test | Selected data can move toward VRAM |
| TI-VRAM-003 | Reuse is measured | Repeated-access test | VRAM working-set hits are real |
| TI-VRAM-004 | Eviction works | VRAM-pressure test | Lower-value data can leave the working set |
| TI-VRAM-005 | VRAM budget is bounded | GPU memory telemetry | TierInfer respects configured/detected limits |
| TI-VRAM-006 | Runtime overhead is protected | Long inference test | Weight management does not starve KV/runtime memory |
| TI-VRAM-007 | CUDA allocation failure degrades safely | Failure test | No corruption or uncontrolled crash |

## A.9 Sparse-MoE and Expert Tracking Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-MOE-001 | Real expert activation can be observed where runtime permits | Runtime instrumentation | Expert IDs originate from valid model/runtime information |
| TI-MOE-002 | Activation frequency is tracked | Multi-token run | Counts match observed routing |
| TI-MOE-003 | Recency is tracked | Runtime telemetry | Recent activity affects recorded state |
| TI-MOE-004 | Hot/cold classification works | Controlled workload | Classification follows measured activity |
| TI-MOE-005 | Expert reuse can influence residency | Runtime test | Repeated experts receive different treatment |
| TI-MOE-006 | No fake expert identity is generated | Code audit + negative test | Unknown mappings remain unknown |

## A.10 Predictor / Prerouter Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-PRED-001 | Predictor is connected to production inference | Execution trace | Real inference causes predictions |
| TI-PRED-002 | Predictions affect decisions | Prefetch/residency trace | Prediction changes actual TierInfer action |
| TI-PRED-003 | Correct predictions are measured | Runtime validation | Accuracy derives from later actual routing |
| TI-PRED-004 | Incorrect predictions are measured | Runtime validation | Incorrect predictions are not hidden |
| TI-PRED-005 | Precision is reported | Benchmark | Metric is calculated from real events |
| TI-PRED-006 | Recall is reported where meaningful | Benchmark | Metric is calculated from real events |
| TI-PRED-007 | Trainable prerouter persists state if required by effective scope | Train/save/load test | Learned state survives restart |
| TI-PRED-008 | Predictor failure has safe fallback | Failure test | Inference remains correct without useful prediction |

## A.11 Async Prefetch Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-PREF-001 | Prefetch is genuinely asynchronous | Timing trace | I/O overlaps other useful work where possible |
| TI-PREF-002 | Prefetch queue is bounded | Stress test | Requests cannot grow without bound |
| TI-PREF-003 | Duplicate requests are handled | Unit/integration test | Redundant I/O is avoided |
| TI-PREF-004 | Useful prefetch is identified correctly | Runtime correlation | Data arrives before demand and is subsequently consumed |
| TI-PREF-005 | Late prefetch is identified | Runtime correlation | Demand occurs before prefetch becomes usable |
| TI-PREF-006 | Unused prefetch is identified | Runtime correlation | Prefetched data never consumed is recorded |
| TI-PREF-007 | Prefetch errors are safe | Failure injection | Failed speculative I/O cannot corrupt inference |
| TI-PREF-008 | Prefetch provides measurable workload value | Native/prefetch comparison | Benefit or cost is quantitatively known |

## A.12 Adaptive Policy Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-POLICY-001 | Policy considers VRAM capacity | Configuration/runtime test | Decisions change with available VRAM |
| TI-POLICY-002 | Policy considers RAM capacity | Configuration/runtime test | Decisions change with available RAM |
| TI-POLICY-003 | Policy considers model size/layout | Multiple-model test | Policy reflects model requirements |
| TI-POLICY-004 | Policy considers activity | Dynamic workload | Hot/cold behavior influences decisions |
| TI-POLICY-005 | Policy considers storage cost | Storage test | I/O characteristics influence policy where appropriate |
| TI-POLICY-006 | Policy considers transfer/prefetch effectiveness | Runtime adaptation test | Ineffective behavior can be reduced |
| TI-POLICY-007 | Manual overrides work | Configuration test | Explicit valid settings are honored |
| TI-POLICY-008 | Defaults are safe | Clean-start validation | System operates without manually calculated tier budgets |

## A.13 Automatic Configuration Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-AUTO-001 | GPU/VRAM detection works | Host inspection | RTX 5090 resources are correctly detected |
| TI-AUTO-002 | System RAM detection works | Host inspection | Available memory is correctly detected |
| TI-AUTO-003 | Model size is detected | Model inspection | Six-shard total is handled correctly |
| TI-AUTO-004 | Storage location/capability is detected sufficiently | 4M2 validation | Tier 2 backend is configured correctly |
| TI-AUTO-005 | Runtime capabilities are detected | llama.cpp + FreeToken tests | Unsupported features are not assumed |
| TI-AUTO-006 | Safety margins are applied | Configuration evidence | Runtime resources are not fully consumed by TierInfer |
| TI-AUTO-007 | Generated configuration is inspectable | CLI/API/telemetry | User can see what TierInfer selected |

## A.14 llama.cpp Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-LLAMA-001 | llama.cpp adapter exists | Code audit | Adapter is production code |
| TI-LLAMA-002 | Adapter is connected to real inference | Runtime trace | TierInfer activity occurs during llama.cpp generation |
| TI-LLAMA-003 | Native llama.cpp remains measurable | Baseline benchmark | Native path can be compared fairly |
| TI-LLAMA-004 | GGUF model layout is integrated | 480B validation | Real GGUF ranges drive TierInfer behavior |
| TI-LLAMA-005 | Sharded model works | Qwen3-Coder run | Six-shard model initializes correctly |
| TI-LLAMA-006 | TierInfer storage path is real | I/O telemetry | TierInfer-generated storage operations are observed |
| TI-LLAMA-007 | RAM tier is real | Cache telemetry | Real cache hits/misses occur |
| TI-LLAMA-008 | VRAM integration is real where runtime permits | GPU telemetry | Residency/movement is observed |
| TI-LLAMA-009 | MoE information is integrated where technically exposed | Expert telemetry | Real routing information is used |
| TI-LLAMA-010 | TierInfer-enabled inference remains correct | Output/runtime validation | No corruption from tier management |
| TI-LLAMA-011 | Native vs TierInfer benchmark exists | Repeated benchmark artifact | Performance contribution is quantitatively known |
| TI-LLAMA-012 | Large-model test passes | 480B validation | Model executes under genuine tier pressure |

## A.15 FreeToken Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-FT-001 | FreeToken adapter exists | Code audit | Adapter is production code |
| TI-FT-002 | Adapter is connected to real FreeToken inference | Runtime trace | TierInfer participates during generation |
| TI-FT-003 | FreeToken-native offload mechanisms are identified | Architecture/runtime audit | TierInfer does not blindly duplicate native mechanisms |
| TI-FT-004 | TierInfer and FreeToken responsibilities are explicit | Code + documentation | Ownership of residency/offload decisions is unambiguous |
| TI-FT-005 | TierInfer NVMe capability integrates where technically applicable | Runtime evidence | Tier 2 contributes to real FreeToken workload |
| TI-FT-006 | RAM behavior is measurable | Runtime telemetry | Native and TierInfer RAM behavior can be distinguished sufficiently |
| TI-FT-007 | VRAM behavior is measurable | GPU telemetry | Relevant residency can be observed |
| TI-FT-008 | MoE/expert integration uses real FreeToken information where available | Runtime evidence | Expert behavior is not fabricated |
| TI-FT-009 | TierInfer telemetry distinguishes native FreeToken behavior | Correlated telemetry | Metrics do not claim native actions as TierInfer actions |
| TI-FT-010 | TierInfer-enabled FreeToken inference remains correct | Runtime validation | No corruption from integration |
| TI-FT-011 | Native vs TierInfer benchmark exists | Repeated benchmark artifact | Benefit/overhead is quantitatively known |
| TI-FT-012 | Failure/fallback behavior is safe | Failure test | FreeToken can continue correctly or fail explicitly |

## A.16 Unified Telemetry Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-TEL-001 | VRAM usage is reported accurately | Correlation with GPU tools | Values are credible |
| TI-TEL-002 | RAM usage is reported accurately | OS correlation | Values are credible |
| TI-TEL-003 | NVMe activity is reported accurately | iostat/storage correlation | Values are credible |
| TI-TEL-004 | Tier hits/misses represent real accesses | Trace correlation | No synthetic hits |
| TI-TEL-005 | Promotions/demotions are recorded | Runtime trace | Events correspond to real transfers |
| TI-TEL-006 | Bytes transferred are recorded | I/O correlation | Accounting matches observed operations |
| TI-TEL-007 | Transfer latency is recorded | Timing evidence | Values derive from actual operations |
| TI-TEL-008 | Prefetch metrics are real | Event correlation | Issued/useful/late/wasted are distinguished |
| TI-TEL-009 | Expert metrics are real | Runtime correlation | Metrics originate from valid expert activity |
| TI-TEL-010 | Performance metrics are captured | Benchmark correlation | TTFT/tokens/s are reproducible |
| TI-TEL-011 | Runtime source is identifiable | llama.cpp/FreeToken tests | Native and TierInfer events are distinguishable |

## A.17 Correctness and Failure-Safety Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-SAFE-001 | Tensor/range data cannot be mixed incorrectly | Integrity tests | Correct data returned |
| TI-SAFE-002 | Eviction cannot race active use | Concurrency test | No use-after-eviction |
| TI-SAFE-003 | Partial async reads cannot become valid cache entries | Failure test | Incomplete data rejected |
| TI-SAFE-004 | VRAM exhaustion is safe | Pressure test | Correct fallback or explicit failure |
| TI-SAFE-005 | RAM exhaustion is safe | Pressure test | Host is not destabilized |
| TI-SAFE-006 | NVMe read failure is safe | Failure injection | No silent corruption |
| TI-SAFE-007 | Wrong prediction is safe | Predictor test | Correct demand path still works |
| TI-SAFE-008 | Late prefetch is safe | Prefetch test | Demand fallback remains correct |
| TI-SAFE-009 | Unsupported feature is explicit | Negative test | No false capability claim |
| TI-SAFE-010 | Silent TierInfer-disable/fallback is detectable | Telemetry test | User knows whether TierInfer is actually active |

## A.18 Qwen3-Coder-480B Large-Model Acceptance Matrix

Designated validation model: Qwen3-Coder-480B-A35B-Instruct, Q4_K_M, 6 GGUF shards, ~290 GB decimal / ~270 GiB, at `/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/`.

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-480B-001 | All six shards exist | File validation | Expected shard set complete |
| TI-480B-002 | Shard sizes are valid | Recorded sizes | No incomplete download |
| TI-480B-003 | llama.cpp recognizes model | Compatibility run | Metadata loads |
| TI-480B-004 | Model initializes | Minimal inference | Runtime reaches inference |
| TI-480B-005 | Stable GPU configuration is identified | Incremental offload test | No uncontrolled CUDA OOM |
| TI-480B-006 | Genuine RAM pressure occurs | Memory telemetry | Model cannot simply remain fully resident in fast memory |
| TI-480B-007 | Active NVMe participation occurs | 4M2 telemetry | Storage reads occur during relevant inference |
| TI-480B-008 | Native baseline is recorded | Benchmark artifact | Native behavior reproducible |
| TI-480B-009 | TierInfer run is recorded | Benchmark artifact | TierInfer behavior reproducible |
| TI-480B-010 | Native/TierInfer I/O is compared | Storage telemetry | Differences quantified |
| TI-480B-011 | Native/TierInfer generation is compared | Repeated benchmark | Throughput/latency differences quantified |
| TI-480B-012 | TierInfer contribution is demonstrated | Combined evidence | Results cannot be explained solely as Linux/md behavior |

## A.19 Performance Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-PERF-001 | Benchmarks are repeatable | Multiple runs | Same configuration produces reasonably consistent results |
| TI-PERF-002 | Cold-cache behavior is measured | Controlled runs | Cold results identified separately |
| TI-PERF-003 | Warm-cache behavior is measured | Controlled runs | Warm results identified separately |
| TI-PERF-004 | Prompt throughput is measured | Runtime metrics | tokens/s recorded |
| TI-PERF-005 | Generation throughput is measured | Runtime metrics | tokens/s recorded |
| TI-PERF-006 | TTFT is measured where available | Runtime metrics | latency recorded |
| TI-PERF-007 | NVMe bytes/token is measured or derivable | Correlated metrics | Storage efficiency known |
| TI-PERF-008 | IOPS/request size is measured | Storage metrics | I/O pattern known |
| TI-PERF-009 | Prefetch benefit/cost is measured | A/B test | Contribution known |
| TI-PERF-010 | Cache benefit/cost is measured | A/B test | Contribution known |
| TI-PERF-011 | Predictor benefit/cost is measured | A/B test | Contribution known |
| TI-PERF-012 | Regressions are retained as evidence | Benchmark history | No selective reporting |

## A.20 FlowRunner Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-FLOW-001 | FlowRunner can identify TierInfer capability | Integration test | Capability exposed cleanly |
| TI-FLOW-002 | FlowRunner can use TierInfer-enabled llama.cpp | End-to-end run | Real inference succeeds |
| TI-FLOW-003 | FlowRunner can use TierInfer-enabled FreeToken | End-to-end run | Real inference succeeds |
| TI-FLOW-004 | FlowRunner does not duplicate tier policy | Code audit | TierInfer owns memory hierarchy |
| TI-FLOW-005 | Runtime selection remains abstracted | Integration test | Flow definition does not require TierInfer internals |

## A.21 Quality Acceptance Matrix

| ID | Requirement | Required Evidence | Acceptance Condition |
|---|---|---|---|
| TI-QUAL-001 | TODO/FIXME/stub audit completed | Audit artifact | Scope-required incomplete work identified |
| TI-QUAL-002 | Production paths traced | Audit artifact | Architecture mapped to executable code |
| TI-QUAL-003 | Fake/synthetic production behavior removed | Code review + tests | Required mechanisms are real |
| TI-QUAL-004 | Disconnected components identified | Audit | No hidden architecture-only components |
| TI-QUAL-005 | Blocking disconnected components repaired | Runtime evidence | Required components participate in execution |
| TI-QUAL-006 | Ignored configuration identified and corrected | Config tests | Exposed configuration works |
| TI-QUAL-007 | Silent errors removed | Failure tests | Errors handled explicitly |

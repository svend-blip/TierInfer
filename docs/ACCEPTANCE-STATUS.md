# Acceptance status — live

One row per Acceptance ID of `docs/ACCEPTANCE-MATRIX.md`. Status per A.2;
evidence points at repository paths, commits, tests, artefacts and
checkpoints in `docs/CHECKPOINTS.md`. Updated at every checkpoint; the
date at the top is the last reconciliation.

**Last reconciled: 2026-09-18 (CP-11: loader's first live A/B on GLM).**

Abbreviations: V = `benchmarks/480b/VALIDATION-480B.md`; CP-n =
`docs/CHECKPOINTS.md`; AUD = `docs/AUDIT-2026-09-18.md`; RO =
`benchmarks/replay-out/480b/`.

## Core (A.4)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-CORE-001 | IN_PROGRESS | RAM and NVMe tiers exist under llama.cpp via the loader (`src/tierinfer/loader.py`, `tools/uffd/`); VRAM under llama.cpp is llama.cpp's own (`-ncmoe`), TierInfer's `VramResidency` cannot be consumed by its kernels — design §"What the VRAM tier is" |
| TI-CORE-002 | VERIFIED | loader's evictions bound llama.cpp's RSS at the budget (29.1 GB vs 33.6 native) during live generation (CP-11) |
| TI-CORE-003 | IMPLEMENTED_UNVERIFIED | loader telemetry: resident set, bytes, evictions, per-token events (`LoaderServer._snapshot_values`, `_token_delta`) |
| TI-CORE-004 | VERIFIED (replay) / IN_PROGRESS (live) | replay arms: hit rates follow routing locality (V §4, code vs prose) |
| TI-CORE-005 | VERIFIED (replay) / IN_PROGRESS (live) | 100 vs 150 GiB tiers under a cgroup; VRAM slots from measured budget (V §4.1) |
| TI-CORE-006 | VERIFIED | page cache dropped behind every read; md0/sda/sdb request sizes 361/403 KB vs native 24/131 KB (V §4.2, §15 method) |
| TI-CORE-007 | IN_PROGRESS | llama.cpp path uses `ExpertCache`, `StorageBackend`, predictors, `Telemetry`; FreeToken integration not yet built |

## Model layout (A.5)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-LAYOUT-001 | ACCEPTED | `tierinfer.gguf`; `tests/test_gguf.py`; real 480B metadata read (CP-3) |
| TI-LAYOUT-002 | ACCEPTED | `read_model`/`shard_paths` (8d33288); `tests/test_shards.py`; six shards, `split.tensors.count=747` matched (CP-3) |
| TI-LAYOUT-003 | ACCEPTED | every shard's last tensor ends at EOF (smoketest goal 2, CP-3); loader test bytes-equal (`tests/test_loader.py`) |
| TI-LAYOUT-004 | ACCEPTED | `ModelIndex` classification; `tierinfer inspect` on 480B (CP-4 notes) |
| TI-LAYOUT-005 | ACCEPTED | 62 MoE layers × 160 experts recognised; routing captured (CP-6a) |
| TI-LAYOUT-006 | ACCEPTED | expert slabs by arithmetic on fused tensors, refuse on non-division; 18 000+ deliveries byte-verified (CP-8b) |
| TI-LAYOUT-007 | ACCEPTED | `GGUFError` on unknown type, non-dividing tensor, missing shard (`tests/test_gguf.py`, `test_index.py`, `test_shards.py`) |

## NVMe tier (A.6)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-NVME-001 | VERIFIED | `StorageBackend`/`ExpertStreamer` preads on index ranges; the loader serves faults from them (`test_loader.py`); 480B replay md0 counters (V §4) |
| TI-NVME-002 | VERIFIED | reads are expert slabs / floor chunks by index (`FileLayout`), 9.5 MB per op (V §4.2) |
| TI-NVME-003 | IN_PROGRESS | `storage.coalesce` exists and is tested; not yet on the loader/prefetch path (item 3) |
| TI-NVME-004 | VERIFIED | demand batching through the streamer (`d55e20e`), measured (V §4.3) |
| TI-NVME-005 | VERIFIED | worker threads + `preadv`, `STREAMING.md` 3.47×; loader prefetch pool |
| TI-NVME-006 | VERIFIED | bounded pool, `pool_exhausted` counted (RO inj2-tiny-pool) |
| TI-NVME-007 | VERIFIED | fail-reads injection: 15 failures → exact path, 0 mismatches (CP-8b) |
| TI-NVME-008 | VERIFIED | V §4.2: merging is md's (24→131 KB native), locality is TierInfer's (361→403 KB) |

## RAM cache (A.7)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-RAM-001 | VERIFIED | `ExpertCache` byte-bounded; 100/150 GiB arms; `used_bytes_end ≤ capacity` (RO summaries) |
| TI-RAM-002 | VERIFIED | admissions counted per arm; loader admits on serve |
| TI-RAM-003 | VERIFIED | replay hits served from held bytes; loader: hit = resident when routed |
| TI-RAM-004 | VERIFIED | misses counted where decided (`897cd37`), 86.8/91.7 % real (V §4) |
| TI-RAM-005 | VERIFIED | 5–7 k evictions per arm; loader eviction test (`test_loader.py`) |
| TI-RAM-006 | VERIFIED | accounting checked (`used_bytes_end` vs capacity; mixed expert sizes `d71d993`) |
| TI-RAM-007 | IMPLEMENTED_UNVERIFIED | locks in loader; concurrency test at scale pending live runs |
| TI-RAM-008 | VERIFIED | expert-granular on GLM and 480B routing |

## VRAM (A.8)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-VRAM-001…005 | VERIFIED (standalone) / BLOCKED (under llama.cpp) | `VramResidency` real `cudaMalloc`/`cudaMemcpy`, 22.1 % hit, budget derived (V §4.1); llama.cpp's kernels cannot consume it without a patch (design) — VRAM under llama.cpp is `-ncmoe` |
| TI-VRAM-006 | VERIFIED | budget leaves KV + reserve + overhead; 131k refused (V subject table) |
| TI-VRAM-007 | VERIFIED | `CudaError` on over-budget admit; tiny-vram 0.3 % hit, 0 mismatches (CP-8b) |

## MoE tracking (A.9)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-MOE-001 | ACCEPTED | `cb_eval` on `ffn_moe_topk`, strided rows verified 97.9 % (CP-4, CP-6a) |
| TI-MOE-002…004 | ACCEPTED | `routing_report.py`: frequency, recency, hot/cold on 480B traces (V §3) |
| TI-MOE-005 | VERIFIED | recency-led retention measured (V §4) |
| TI-MOE-006 | ACCEPTED | identities only from the router; `owner_of_page` never invents a key |

## Predictor (A.10)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-PRED-001 | IMPLEMENTED_UNVERIFIED | loader `_prefetch_after` uses the predictor on live routing; runtime evidence pending |
| TI-PRED-002 | VERIFIED (replay) | prefetch issued by prediction (RO pf-d8) |
| TI-PRED-003…006 | VERIFIED | recall@k, waste, per arm (V §5, RO) |
| TI-PRED-007 | VERIFIED (tests) / IN_PROGRESS (eval) | `prerouter.py`: per-layer linear multi-label, online SGD, `.npz` persistence; save/load survives restart (`test_prerouter.py`); recall on 480B traces pending an idle machine |
| TI-PRED-008 | VERIFIED | bad-predictor injection, 0 mismatches (CP-8b) |

## Prefetch (A.11)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-PREF-001 | IN_PROGRESS | asynchronous by construction, priority classes + coalescing (`PrefetchScheduler`, `_serve_run`); the 480B A/B ran at depth 0 — depth>0 arms under the loader pending |
| TI-PREF-002 | VERIFIED | `BufferPool`, `pool_exhausted` |
| TI-PREF-003 | VERIFIED | in-flight/resident dedup (`Prefetcher.before_layer`, loader `_serving`) |
| TI-PREF-004…006 | VERIFIED | useful/late/wasted split (`8d33288`; RO) |
| TI-PREF-007 | VERIFIED | fail-reads (CP-8b) |
| TI-PREF-008 | VERIFIED (replay: none) / IN_PROGRESS (live) | V §4.2 negative result; loader A/B pending |

## Policy (A.12)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-POLICY-001…003 | VERIFIED | `autoconfig` table (V subject); `tests/test_autoconfig.py` |
| TI-POLICY-004 | VERIFIED | recency-led retention |
| TI-POLICY-005 | VERIFIED | `probe_concurrency` decides demand batching per device (`5ceaf53`) |
| TI-POLICY-006 | IN_PROGRESS | `TierPolicy` dial is a simulator (AUD 9); live adaptation of prefetch depth not yet wired into the loader |
| TI-POLICY-007 | VERIFIED | `--ram-gb`, `--depth`, `--batch-demand` honoured (`tests/test_autoconfig.py`, replay) |
| TI-POLICY-008 | IMPLEMENTED_UNVERIFIED | `tierinfer serve` defaults to autoconfig's share; live default run pending |

## Autoconfig (A.13)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-AUTO-001…003 | ACCEPTED | `Host.measure`, six-shard size (CP-4 notes) |
| TI-AUTO-004 | VERIFIED | concurrency probe on the 4M2; fuller device characteristics = item 4 |
| TI-AUTO-005 | IN_PROGRESS | llama.cpp capabilities known; FreeToken pending |
| TI-AUTO-006 | ACCEPTED | reserve measured, KV, 512 MB overhead, 60 % RAM share |
| TI-AUTO-007 | ACCEPTED | `Configuration.explain()`, `tierinfer inspect` |

## llama.cpp (A.14)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-LLAMA-001 | IMPLEMENTED_UNVERIFIED | `tools/uffd/tierinfer_mmap.c` + `tierinfer.loader` (`f417eed`) |
| TI-LLAMA-002 | VERIFIED | GLM live run: faults, copies, evictions during generation (CP-11 telemetry) |
| TI-LLAMA-003 | ACCEPTED | native runs without the shim (CP-4, CP-5) |
| TI-LLAMA-004 | VERIFIED | `FileLayout` per shard drove every fault of the 480B runs (30 321 regions over six files; CP-12) |
| TI-LLAMA-005 | VERIFIED | six-shard 480B served through the loader, three runs, identical tokens (CP-12) |
| TI-LLAMA-006 | VERIFIED | loader's preads observed on nvme0n1p2 during generation (CP-11) |
| TI-LLAMA-007 | VERIFIED | 72 % resident hits per generated token from the loader's own residency set (CP-11) |
| TI-LLAMA-008 | BLOCKED | VRAM under llama.cpp is llama.cpp's (`-ncmoe`); a TierInfer VRAM tier needs a llama.cpp patch — documented, not attempted |
| TI-LLAMA-009 | VERIFIED | shim `cb_eval` → ROUTE: 21 194 routed experts scored hit/miss on the 480B run (telemetry, CP-12) |
| TI-LLAMA-010 | VERIFIED | 32 greedy tokens identical to native (CP-11) |
| TI-LLAMA-011 | VERIFIED | 480B, three runs each: loader 0.647 t/s (0.622–0.668) vs native 0.467 (0.437–0.481), identical tokens, 36 % fewer bytes (CP-12, V §4.4) |
| TI-LLAMA-012 | VERIFIED | 480B under genuine tier pressure: 150 GB tier for 9 920 experts (~50 %), full after the prompt batch, 885 evictions, three runs 0.622/0.668/0.647 t/s (CP-12) |

## FreeToken (A.15)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-FT-001 | IMPLEMENTED_UNVERIFIED | `adapters/freetoken.py` — configuration in, counters out (AUD 13) |
| TI-FT-002 | IMPLEMENTED_UNVERIFIED | FreeToken patch `patches/freetoken-tierinfer-tier.patch` (`HostResidency.TIERED`, `HostBank(backing="tierinfer")`, CPU-executor `ROUTED`); runtime trace on Flash-Next pending |
| TI-FT-003 | VERIFIED | design spec + `docs/FREETOKEN.md`: slot cache (VRAM) and pinned/locked banks (RAM) are FreeToken's; TierInfer only serves banks FreeToken would otherwise fill at load |
| TI-FT-004 | VERIFIED | ownership explicit in code and docs: `pin()` refuses a tiered bank, GPU layers untouched, tier applies to `--moe-cpu-layers` only |
| TI-FT-005 | IN_PROGRESS | `tierinfer serve <ftw-dir>` + `tierinfer.client` built and tested on a synthetic checkpoint; Flash-Next run pending |
| TI-FT-006 | IN_PROGRESS | loader telemetry (faults, bytes, hit rate by faults, evictions) per tiered layer; FreeToken's `/v1/stats` beside it in `freetoken_ab.py` |
| TI-FT-007 | IN_PROGRESS | FreeToken's slot cache stats and VRAM from `/v1/stats`; TierInfer holds no VRAM under FreeToken by design |
| TI-FT-008 | IMPLEMENTED_UNVERIFIED | real `topk_ids` from the CPU executor's pinned log → `ROUTED` |
| TI-FT-009 | IN_PROGRESS | separate namespaces: `loader.*` vs FreeToken's own stats in the run record |
| TI-FT-010 | IN_PROGRESS | greedy output compared across arms by `freetoken_ab.py`; synthetic rows verified byte-exact |
| TI-FT-011 | IN_PROGRESS | `benchmarks/freetoken_ab.py` ready; needs the GPU (480B harness running) |
| TI-FT-012 | IN_PROGRESS | `TIERINFER_SOCK` unset → unpatched behaviour; tier absent → explicit `RuntimeError`; refused evictions counted; failure injections under FreeToken pending |

## Telemetry (A.16)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-TEL-001 | VERIFIED | `nvidia-smi` sampled alongside (`sample.py`), VRAM after load per sweep row |
| TI-TEL-002 | VERIFIED | `/proc/meminfo` sampled; RSS per run |
| TI-TEL-003 | VERIFIED | `/proc/diskstats` md0 + members, per token |
| TI-TEL-004 | VERIFIED | after `897cd37` |
| TI-TEL-005 | IN_PROGRESS | evictions recorded; promotions under the loader = serves |
| TI-TEL-006…007 | VERIFIED | bytes, read seconds, copy seconds |
| TI-TEL-008 | VERIFIED | issued/useful/late/wasted |
| TI-TEL-009 | ACCEPTED | routing from `cb_eval` |
| TI-TEL-010 | VERIFIED | llama-server timings (prompt/gen t/s), TTFT via `first_token` in the trace tool |
| TI-TEL-011 | IN_PROGRESS | `sim.*` split from observed; native vs loader runs labelled |

## Safety (A.17)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-SAFE-001 | VERIFIED | byte-verified deliveries (CP-8b), loader tests |
| TI-SAFE-002 | IMPLEMENTED_UNVERIFIED | loader evicts interior pages only, per-key serving lock; a page evicted mid-use faults again and is re-served — correct by construction (uffd), concurrency test at scale pending |
| TI-SAFE-003 | VERIFIED | short read raises before recording (`8d33288`) |
| TI-SAFE-004 | VERIFIED | tiny-vram; CUDA OOM in the sweep was llama.cpp's, reported |
| TI-SAFE-005 | VERIFIED | cgroup-scoped arms; host never destabilised |
| TI-SAFE-006 | VERIFIED | fail-reads |
| TI-SAFE-007…008 | VERIFIED | bad-predictor; late counted and served |
| TI-SAFE-009 | VERIFIED | refusals in autoconfig/index; shim "standing aside" message |
| TI-SAFE-010 | VERIFIED | shim reports standing aside / serving on stderr; server logs each mapping; telemetry `mapping` event (CP-11) |

## 480B (A.18)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-480B-001…005 | ACCEPTED | CP-3, CP-5 |
| TI-480B-006…008 | ACCEPTED | CP-4 (183 GB available vs 270 GiB; 1.8 GB/token from md0 during generation) |
| TI-480B-009…010 | VERIFIED | replay V §4.2; live loader V §4.4 |
| TI-480B-011 | VERIFIED | live A/B, three runs per arm, V §4.4 |
| TI-480B-012 | VERIFIED | replay V §4.2 + live loader V §4.4 (three runs) |

## Performance (A.19)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-PERF-001…008 | VERIFIED | CP-4/5 tables, RO |
| TI-PERF-009…010 | VERIFIED (replay) | V §4 |
| TI-PERF-011 | IN_PROGRESS | predictor A/B under the loader pending |
| TI-PERF-012 | ACCEPTED | negative results kept (V §4.2, §4.3) |

## FlowRunner (A.20)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-FLOW-001…005 | NOT_STARTED | item 5; `adapters/flowrunner.py` schema exists, no consumer in FlowRunner |

## Quality (A.21)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-QUAL-001…004 | ACCEPTED | AUD |
| TI-QUAL-005 | IN_PROGRESS | shards repaired; loader is the repair for item 12 of AUD |
| TI-QUAL-006 | VERIFIED | assist flags removed; `batch_demand` honoured |
| TI-QUAL-007 | VERIFIED | fadvise failures raised; timeouts guarded |

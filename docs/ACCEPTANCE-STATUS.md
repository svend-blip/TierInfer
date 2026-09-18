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
| TI-CORE-001 | VERIFIED | RAM and NVMe tiers exist under llama.cpp via the loader (`src/tierinfer/loader.py`, `tools/uffd/`); VRAM under llama.cpp is llama.cpp's own (`-ncmoe`), TierInfer's `VramResidency` cannot be consumed by its kernels — design §"What the VRAM tier is" |
| TI-CORE-002 | VERIFIED | loader's evictions bound llama.cpp's RSS at the budget (29.1 GB vs 33.6 native) during live generation (CP-11) |
| TI-CORE-003 | VERIFIED | per-token events and run records under llama.cpp (CP-12) and FreeToken (CP-13); read back by FlowRunner (`benchmarks/flowrunner-out/`) |
| TI-CORE-004 | VERIFIED (replay) / IN_PROGRESS (live) | replay arms: hit rates follow routing locality (V §4, code vs prose) |
| TI-CORE-005 | VERIFIED (replay) / IN_PROGRESS (live) | 100 vs 150 GiB tiers under a cgroup; VRAM slots from measured budget (V §4.1) |
| TI-CORE-006 | VERIFIED | page cache dropped behind every read; md0/sda/sdb request sizes 361/403 KB vs native 24/131 KB (V §4.2, §15 method) |
| TI-CORE-007 | VERIFIED | one `LoaderServer` (cache, storage, predictors, telemetry) serves llama.cpp (shim) and FreeToken (`tierinfer.client`) alike (CP-12, CP-13) |

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
| TI-NVME-003 | VERIFIED | coalesced runs on the prefetch path (`_serve_run`): 33 coalesced reads for 22 experts in the FlowRunner GLM run |
| TI-NVME-004 | VERIFIED | demand batching through the streamer (`d55e20e`), measured (V §4.3) |
| TI-NVME-005 | VERIFIED | worker threads + `preadv`, `STREAMING.md` 3.47×; loader prefetch pool |
| TI-NVME-006 | VERIFIED | bounded pool, `pool_exhausted` counted (RO inj2-tiny-pool) |
| TI-NVME-007 | VERIFIED | fail-reads injection: 15 failures → exact path, 0 mismatches (CP-8b) |
| TI-NVME-008 | VERIFIED | V §4.2: merging is md's (24→131 KB native), locality is TierInfer's (361→403 KB); `--align 524288` under the loader: 10 % fewer requests, 15 % more bytes, t/s within noise (CP-15) |

## RAM cache (A.7)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-RAM-001 | VERIFIED | `ExpertCache` byte-bounded; 100/150 GiB arms; `used_bytes_end ≤ capacity` (RO summaries) |
| TI-RAM-002 | VERIFIED | admissions counted per arm; loader admits on serve |
| TI-RAM-003 | VERIFIED | replay hits served from held bytes; loader: hit = resident when routed |
| TI-RAM-004 | VERIFIED | misses counted where decided (`897cd37`), 86.8/91.7 % real (V §4) |
| TI-RAM-005 | VERIFIED | 5–7 k evictions per arm; loader eviction test (`test_loader.py`) |
| TI-RAM-006 | VERIFIED | accounting checked (`used_bytes_end` vs capacity; mixed expert sizes `d71d993`) |
| TI-RAM-007 | VERIFIED | 32 compute threads faulting against 8 workers over 59 k faults per 480B run, 0 repeat faults, tokens identical (CP-12) |
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
| TI-PRED-001 | VERIFIED | live: 271 prefetches from the predictor on real ROUTE in the FlowRunner GLM run (170 useful, 126 late, 93 wasted) |
| TI-PRED-002 | VERIFIED (replay) | prefetch issued by prediction (RO pf-d8) |
| TI-PRED-003…006 | VERIFIED | recall@k, waste, per arm (V §5, RO) |
| TI-PRED-007 | VERIFIED | `prerouter.py` persists (`test_prerouter.py`); on 480B traces online recall@16 77.4 % prose / 66.3 % code, +10 points over the adaptive blend (V §5.1) |
| TI-PRED-008 | VERIFIED | bad-predictor injection, 0 mismatches (CP-8b) |

## Prefetch (A.11)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-PREF-001 | VERIFIED | depth 8 under the loader: 74 % of guesses landed before use (asynchronous, overlapping); generation unchanged vs depth 0 — the overlap buys nothing on a saturated device (CP-15) |
| TI-PREF-002 | VERIFIED | `BufferPool`, `pool_exhausted` |
| TI-PREF-003 | VERIFIED | in-flight/resident dedup (`Prefetcher.before_layer`, loader `_serving`) |
| TI-PREF-004…006 | VERIFIED | useful/late/wasted split (`8d33288`; RO) |
| TI-PREF-007 | VERIFIED | fail-reads (CP-8b) |
| TI-PREF-008 | VERIFIED (negative) | replay V §4.2; live under FreeToken: depth 8 → 48 guesses in 62 steps, 14 useful, 6.3 vs 6.1 t/s (CP-13) |

## Policy (A.12)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-POLICY-001…003 | VERIFIED | `autoconfig` table (V subject); `tests/test_autoconfig.py` |
| TI-POLICY-004 | VERIFIED | recency-led retention |
| TI-POLICY-005 | VERIFIED | `probe_concurrency` decides demand batching per device (`5ceaf53`) |
| TI-POLICY-006 | VERIFIED | `serve --adapt-depth`: depth re-decided every 8 tokens from yield (`tests/test_loader_policy.py`); live value bounded by prefetch not paying on this device (CP-15) |
| TI-POLICY-007 | VERIFIED | `--ram-gb`, `--depth`, `--batch-demand` honoured (`tests/test_autoconfig.py`, replay) |
| TI-POLICY-008 | VERIFIED | the FlowRunner run's tier and depth came from `tierinfer resolve` (capability + host), not from flags |

## Autoconfig (A.13)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-AUTO-001…003 | ACCEPTED | `Host.measure`, six-shard size (CP-4 notes) |
| TI-AUTO-004 | VERIFIED | concurrency probe on the 4M2; fuller device characteristics = item 4 |
| TI-AUTO-005 | VERIFIED | llama.cpp: shim + cb_eval; FreeToken: tiered CPU-executor layers, pinned GPU layers refused (`docs/FREETOKEN.md`) |
| TI-AUTO-006 | ACCEPTED | reserve measured, KV, 512 MB overhead, 60 % RAM share |
| TI-AUTO-007 | ACCEPTED | `Configuration.explain()`, `tierinfer inspect` |

## llama.cpp (A.14)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-LLAMA-001 | VERIFIED | the loader under llama.cpp on GLM and the 480B (CP-11, CP-12) |
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
| TI-FT-001 | VERIFIED | `adapters/freetoken.py` + `tierinfer.client` + the FreeToken patch, run on Flash-Next (CP-13) |
| TI-FT-002 | VERIFIED | Flash-Next runs: 12 tiered layers served during generation, 58 k faults, 20 GB (CP-13) |
| TI-FT-003 | VERIFIED | design spec + `docs/FREETOKEN.md`: slot cache (VRAM) and pinned/locked banks (RAM) are FreeToken's; TierInfer only serves banks FreeToken would otherwise fill at load |
| TI-FT-004 | VERIFIED | ownership explicit in code and docs: `pin()` refuses a tiered bank, GPU layers untouched, tier applies to `--moe-cpu-layers` only |
| TI-FT-005 | VERIFIED | the tier serves 17.1 GB of banks under 16 GB / 8 GB budgets on Flash-Next; decode at native speed when the budget holds the layers (CP-13) |
| TI-FT-006 | VERIFIED | per-step hits/misses/bytes/evictions from TierInfer beside FreeToken's `/v1/stats` (CP-13 table) |
| TI-FT-007 | VERIFIED | FreeToken's `vram_bytes`/slot cache in the run record; TierInfer holds no VRAM under FreeToken by design (CP-13) |
| TI-FT-008 | VERIFIED | real `topk_ids` (device→host log, stream drained) → `ROUTED`; 85.8 % hit rate consistent with bytes copied (CP-13) |
| TI-FT-009 | VERIFIED | `loader.*` vs FreeToken's `stats_before/after` in every run JSON (CP-13) |
| TI-FT-010 | VERIFIED | greedy output identical across eleven runs, native and tiered (CP-13) |
| TI-FT-011 | VERIFIED | `benchmarks/freetoken-out/flashnext.md`: native ×2, tiered 8/16 GB, prefetch; repeated (CP-13) |
| TI-FT-012 | VERIFIED | live: `tierinfer serve` killed 12 s into a completion; the client served its own faults, output identical to native (CP-16); unset socket → unpatched path; absent tier → explicit error |

## Telemetry (A.16)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-TEL-001 | VERIFIED | `nvidia-smi` sampled alongside (`sample.py`), VRAM after load per sweep row |
| TI-TEL-002 | VERIFIED | `/proc/meminfo` sampled; RSS per run |
| TI-TEL-003 | VERIFIED | `/proc/diskstats` md0 + members, per token |
| TI-TEL-004 | VERIFIED | after `897cd37` |
| TI-TEL-005 | VERIFIED | serves (faults, bytes) and evictions per token in every live run's telemetry |
| TI-TEL-006…007 | VERIFIED | bytes, read seconds, copy seconds |
| TI-TEL-008 | VERIFIED | issued/useful/late/wasted |
| TI-TEL-009 | ACCEPTED | routing from `cb_eval` |
| TI-TEL-010 | VERIFIED | llama-server timings (prompt/gen t/s), TTFT via `first_token` in the trace tool |
| TI-TEL-011 | VERIFIED | `sim.*` split; native vs loader vs FreeToken-native labelled per run record; FlowRunner tags `engine` |

## Safety (A.17)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-SAFE-001 | VERIFIED | byte-verified deliveries (CP-8b), loader tests |
| TI-SAFE-002 | VERIFIED (885–4 316 evictions per live run, tokens identical) | loader evicts interior pages only, per-key serving lock; a page evicted mid-use faults again and is re-served — correct by construction (uffd), concurrency test at scale pending |
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
| TI-PERF-011 | VERIFIED (negative) | prerouter vs blend at depth 8 under the loader: 0.19 vs 0.35 t/s — better recall, five times the speculative reads, half the speed (CP-15) |
| TI-PERF-012 | ACCEPTED | negative results kept (V §4.2, §4.3) |

## FlowRunner (A.20)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-FLOW-001 | VERIFIED | `flowrunner engine` resolves the capability through `tierinfer resolve` (FlowRunner `internal/tierinfer`) |
| TI-FLOW-002 | VERIFIED | real run on GLM-4.5-Air: endpoint, completion, timings, `loader.*` telemetry (`benchmarks/flowrunner-out/`, FlowRunner `docs/ENGINE-tierinfer.md`) |
| TI-FLOW-003 | VERIFIED | `runtime: freetoken`: FlowRunner started `tierinfer serve` + `ft serve` on Flash-Next, 31 tokens, TierInfer telemetry read back (`benchmarks/flowrunner-out/ft-engine.out`, FlowRunner `d556ef4`) |
| TI-FLOW-004 | VERIFIED | the adapter passes a capability document and reads telemetry back; no tier decision lives in FlowRunner |
| TI-FLOW-005 | VERIFIED | the flow names model, context and residency share — no shim, socket or tier size |

## Quality (A.21)

| ID | Status | Evidence / blocker |
|---|---|---|
| TI-QUAL-001…004 | ACCEPTED | AUD |
| TI-QUAL-005 | VERIFIED | every blocking audit finding repaired and re-measured (CP-2, CP-12) |
| TI-QUAL-006 | VERIFIED | assist flags removed; `batch_demand` honoured |
| TI-QUAL-007 | VERIFIED | fadvise failures raised; timeouts guarded |

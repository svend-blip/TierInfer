# Qwen3-Coder-480B-A35B on the OWC 4M2 — what TierInfer does, measured

The deliverable `docs/SCOPE-ADDENDUM-480B.md` §23 asks for. Every number
here was observed on this host on 2026-09-18; the raw artefacts (server
timings, `/proc/diskstats` samples every 2 s, replay telemetry) are in
`raw/`, and `docs/CHECKPOINTS.md` records the order things happened in.
Sections still being measured say so.

## Subject

| | |
|---|---|
| Model | Qwen3-Coder-480B-A35B-Instruct, `qwen3moe`, 62 MoE layers, 160 experts, 8 used per token, no shared expert |
| Quantization | Q4_K_M (`general.file_type=15`): gate/up Q4_K, down Q6_K, norms F32 |
| Size | 6 shards, 290 058 826 208 B = 270.1 GiB; routed experts 263.3 GB, everything else 6.8 GB |
| One expert | 30.6 MB (8.85 + 8.85 + 12.9); 496 per token = **20.9 GB working set per token, 7.7 % of the file** (`tierinfer inspect`) |
| Storage | `/dev/md0` RAID0 of two OWC Express 4M2 members (`sda`, `sdb`, USB), 512 k chunk, ext4, `read_ahead_kb=4096`, `max_sectors_kb=512` |
| Host | 32 threads, 187 GiB RAM (183 free during the runs), RTX 5090 32 GB, driver 580.173.02, CUDA 13.0.3, kernel 7.0.0-31 |
| llama.cpp | b10482 (`8b8640097`), `~/llama.cpp-qwen38/build` — `llama-server` for the native baseline, `libllama` behind `tools/trace` for routing capture |
| TierInfer | `main` at the revision `docs/CHECKPOINTS.md` names per checkpoint |

What `autoconfig` derives for this model on this host (reserve = VRAM measured
in use, RAM share 60 % of available minus the floor):

| context | VRAM experts | RAM experts | resident share | KV cache | verdict |
|--:|--:|--:|--:|--:|---|
| 4 096 | 681 (19.4 GiB) | 3 606 (102.8 GiB) | 46.4 % | 0.97 GiB | usable |
| 16 384 | 579 (16.5 GiB) | 3 604 | 45.3 % | 3.88 GiB | usable |
| 65 536 | 171 (4.9 GiB) | 3 604 | 40.8 % | 15.5 GiB | refused: 171 < one token's 496 |
| 131 072 | 0 | 3 604 | 39.0 % | 31.0 GiB | refused: KV alone exceeds the card |

## 1. Native llama.cpp, `-ngl 0` (Linux mmap, page faults, md0)

Fixed 111-token prompt, 32 generated tokens, context 4 096, 32 threads,
`--no-warmup`, flash attention off. "cold" = all six shards evicted with
`posix_fadvise(DONTNEED)` after `fsync` (residency measured 0.0 % before
start); "warm" = started immediately after the previous run with nothing
evicted (the page cache can hold at most ~180 GB of the 290 GB, so warm is
partial by construction). Load time = seconds from process start to
`/health` reporting the model loaded; llama.cpp maps with `MAP_POPULATE`,
which reads the file through once.

| run | load s | prompt t/s | gen t/s | gen: md0 GB/token | md0 reads/token | md0 mean KB | sda mean KB | await ms | md0 util | CPU busy | server RSS GB | major faults |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cold1 | 191 | 0.745 | 0.241 | 1.82 | 73 142 | 25 | 134 | 4.43 | 0.46 | 0.84 | — | — |
| cold2 | 183 | 0.833 | 0.220 | 1.77 | 74 098 | 23 | 125 | 4.24 | 0.43 | 0.85 | 183 | 1 027 458 |
| cold3 | 201 | 0.741 | 0.244 | 1.86 | 78 315 | 24 | 131 | 4.37 | 0.48 | 0.84 | 170 | 190 276 |
| warm1 † | 0 | 2.084 | 0.225 | 1.36 | 63 985 | 22 | 98 | 3.19 | 0.39 | 0.89 | — | — |
| warm2 | 193 | 1.079 | 0.201 | 2.29 | 97 557 | 22 | 147 | 4.84 | 0.48 | 0.83 | 173 | 247 601 |
| warm3 | 171 | 0.767 | 0.196 | 2.18 | 84 011 | 26 | 126 | 4.35 | 0.43 | 0.84 | 183 | 416 553 |
| **cold median (n=3)** | 191 | 0.745 | **0.241** | **1.82** | 74 098 | 24 | 131 | 4.37 | 0.46 | 0.84 | | |
| **warm median (n=3)** | 171 | 1.079 | **0.201** | **2.18** | 84 011 | 22 | 126 | 4.35 | 0.43 | 0.84 | | |

† warm1 reused cold1's still-running server (see the incident in
`docs/CHECKPOINTS.md` CP-3), so it is the one run whose model was *already
mapped*: prompt processing at 2.08 t/s against 0.75–1.08 everywhere else
shows what the mapping's populated pages are worth. RSS and faults were not
captured for the first two runs (the runner read them from the wrong
process); the server's RSS is the mapped file, not an allocation.

Generation ranges 0.196–0.244 t/s over six runs and the "warm" runs are not
faster: a warm start that reloads the server re-populates the mapping from
the file's beginning and evicts what the previous run left, so it reads
*more* per token (2.2 GB) than a cold start (1.8 GB). Only the run that kept
its server was warm in any useful sense, and only for the prompt.

The first thing the table says: native generation is **not bandwidth-bound**.
md0 is under half utilised, delivering ~0.4 GB/s of a device measured at
2 GB/s, while the CPU is 85 % busy — busy faulting. Each token issues ~70 000
reads of ~23 KB (logical) which md merges into ~125 KB physical requests
before the drives see them: the merging is real, it is a factor of five, and
it is Linux's and md's, not anyone else's. The page cache is doing most of
the tier work already: a token needs 20.9 GB and native reads 1.8 GB of it,
so ~90 % of what a token touches was still resident from recent tokens.

## 2. GPU offload point (`-ngl 99 -ncmoe N`)

Same prompt, tokens and context as §1; `-ngl 99` puts every non-expert
tensor on the card and `-ncmoe N` keeps the fused expert tensors of the
first N layers on the CPU, so 62−N layers' experts (4.67 GB each) go to
VRAM. Flash attention on. One cold and one warm run per step; the sweep
stops at the first step the runtime cannot allocate.

| `-ncmoe` | expert layers on GPU | VRAM after load | load s | prompt t/s | gen t/s cold / warm | gen: md0 GB/token | md0 reads/token | md0 util | CPU |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 62 | 0 | 12.9 GB | 193 / 149 | 0.760 / 0.781 | 0.419 / 0.461 | 1.77 / 1.39 | 67 535 / 53 496 | 0.64 / 0.57 | 0.74 |
| 60 | 2 | 22.3 GB | 191 / 147 | 0.768 / 0.899 | 0.463 / 0.380 | 1.51 / 1.67 | 57 763 / 84 508 | 0.59 / 0.62 | 0.72 |
| 58 | 4 | 31.6 GB | 193 / 165 | 0.801 / 0.879 | 0.509 / 0.447 | 1.33 / 1.56 | 50 812 / 58 624 | 0.58 / 0.57 | 0.76 |
| 57 | 5 | — | — | — | **OOM**: `cudaMalloc failed: out of memory` allocating a 29.8 GiB CUDA0 buffer | | | | |

**Selected: `-ngl 99 -ncmoe 60`** — 22.3 GB of 32.6 in use, 9 GB of headroom
for a longer context or a second process; `-ncmoe 58` runs but leaves
0.97 GB, which the next KV allocation would take. The reason it is a
*baseline* and not a *result*: moving attention to the GPU lifts generation
from 0.24 to ~0.42–0.46 t/s, and each further expert layer on the card buys
what its 4.67 GB of experts no longer have to be faulted in — but the
run-to-run spread (0.38–0.51) is as large as the step-to-step gain, because
storage still binds. md0 reads per token fall with the layers moved (67 k →
51 k cold) and utilisation rises to ~0.6, which is the same story as §1 with
less CPU in the way.


## 3. Routing captured from the running 480B

`tools/trace/tierinfer-trace -m <shard 1> -ngl 99 --cpu-moe -t 32 -c 4096 -n 400 -f <prompt>`
— attention on the GPU, the fused expert tensors kept on the CPU by
tensor-buffer override, routing read from `ffn_moe_topk-<layer>` through
`cb_eval` row by row (the strided-view fix, verified on GLM at 97.9 %
agreement between batched and one-at-a-time decodes). Nothing in llama.cpp
is patched. Expert identities are the router's own top-k indices, never
inferred from file access.

| | prose prompt | code prompt |
|---|--:|--:|
| generated tokens (+ prompt) | 400 (+111) | 400 (+151) |
| capture rate | load 187 s, prompt 0.93 t/s, **gen 0.72 t/s** | load 168 s, prompt 1.12 t/s, **gen 0.39 t/s** |
| experts per token | 496 = layout → 20.9 GB, 7.7 % of the file | 496 = layout |
| distinct experts in 400 tokens | 7 121 of 9 920 (71.8 %); 2 799 never routed | 9 011 (90.8 %); 909 never routed |
| activation skew | top 10 % of a layer's experts take **56 %** of its activations | **40 %** |
| neighbour-token overlap | 38.5 % | 26.0 % |
| horizon W=8 / 32 / 128 | 62 GB / 115 GB / **173 GB** (23 / 43 / 64 % of the file) | 73 GB / 147 GB / **221 GB** (27 / 54 / 82 %) |

The horizon row explains §1's native numbers directly: 128 consecutive
prose tokens need 173 GB, the page cache holds about 180 GB, so Linux
already serves ~90 % of a token's 20.9 GB from RAM and reads 1.8 GB. Any RAM
tier TierInfer runs with less than that is starting from behind; any
advantage has to come from *what* is kept and *how* the misses are read, not
from keeping more.

**The prompt class changes the workload by a factor of two.** Code routing
is far less local: 91 % of all experts are touched within 400 tokens against
72 %, neighbouring tokens share 26 % of their experts against 38 %, and 128
tokens need 221 GB — more than the page cache — so the same runtime on the
same storage generated at 0.39 t/s against 0.72. Addendum §16's warning
that routing alters the workload is not hypothetical on this model; every
comparison below is made within a prompt class.

Hot/cold is measurable and strong — 56 % of a layer's activations land on
its 16 most-used experts, and a quarter of all experts were never asked for
in 400 tokens — but §5 shows that frequency alone is the weakest predictor
of the *next* token's experts, as it was on GLM.

## 4. TierInfer replay — real routing, real files, real caches, no compute

_(CP-6/7/8: cache-only, prefetch depth 8/16, emulated compute, VRAM tier,
150 GB RAM tier, page cache allowed; per arm: cache hit rate, prefetch
useful/late/wasted/stalls, md0 GB and reads per token against native, sda
request sizes, VRAM hit rate and transfer rate)_

## 5. Predictor on 480B routing

`benchmarks/routing_report.py`, generated tokens only, 50 warm-up tokens,
each layer predicted before its routing is observed and with the current
token's lower layers available (what a prefetcher would actually know).
All four predictors are **heuristic**; there is no trainable prerouter
(audit item 7).

| recall@k, prose | frequency | persistence | transition | adaptive blend | adaptive waste |
|---|--:|--:|--:|--:|--:|
| k = 8 (one token's 8) | 34.2 % | 38.5 % | **47.0 %** | 46.4 % | 53.6 % |
| k = 16 | 51.9 % | 51.8 % | **66.8 %** | 66.1 % | 67.0 % |
| k = 32 | 72.6 % | 54.4 % | **84.3 %** | 83.4 % | 79.1 % |

Context beats the frequency floor by 13–15 points at k=8–16, as on GLM
(REAL-ROUTING.md); the transition table — which expert follows which across
adjacent layers within the same token — is the strongest single signal here,
and the adaptive blend tracks it to within a point without being told.
Waste is the price: at k=16, two of three prefetched experts go unused.

| recall@k, code | frequency | persistence | transition | adaptive blend | adaptive waste |
|---|--:|--:|--:|--:|--:|
| k = 8 | 24.1 % | 27.1 % | 38.6 % | **38.8 %** | 61.2 % |
| k = 16 | 37.6 % | 39.7 % | **56.6 %** | 55.6 % | 72.2 % |
| k = 32 | 55.6 % | 44.1 % | **74.7 %** | 73.6 % | 81.6 % |

Ten points lower across the board on code, in step with the weaker locality
above; the ordering of the signals is the same. Prediction here is a
statement about which experts to *have ready*; §4 measures what that buys
against real reads.

## 6. Failure behaviour

_(CP-8: always-wrong predictor, injected read failures, two-slot pool,
eight-slot VRAM tier, unresolvable expert — every delivery verified against
an exact read)_

## 7. What could not be measured, and why

**Tokens per second of "llama.cpp + TierInfer".** No loader exists that puts
TierInfer's buffers under llama.cpp's `ggml_mul_mat_id`; the audit names the
point and `SCOPE.md` goal 6 carries it as the remaining work. Every TierInfer
number in this document is I/O and residency under *replayed* real routing;
none is a decode rate, and none is presented as one.

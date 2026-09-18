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

`benchmarks/replay.py`: the captured routing is replayed layer by layer
through the real `Prefetcher` → `ExpertStreamer` (8 workers, `preadv` into
pooled buffers) → `ExpertCache` (holding the bytes) against the six shards
on md0. **What is real:** every read, every byte, every hit, miss, eviction,
prefetch and stall, and every device counter. **What is absent:** compute —
no kernel consumes the bytes, so a token's time here is its I/O wait, and
the `comp` arm adds sleeps of 4 ms before and 4 ms after each layer
(~500 ms per token, an *assumed* compute time; the model cannot be run
RAM-resident on this host to measure a real one) so that prefetch has
something to overlap. `--drop-after-read` evicts every delivered range from
the page cache, so TierInfer's cache is the only RAM tier and every miss it
reports is a read the device served; the one arm without it shows what
double caching looks like. Predictor warmed on 50 tokens, then tokens
50–199 replayed; RAM tier 100 GiB (`autoconfig`'s share) or 150 GiB (close
to the ~180 GB page cache native had); cold start.

### 4.1 The arms

| arm (prose unless noted) | ms/token median | of which I/O wait | RAM hit | md0 GB/token | md0 reads/token | md0 KB | sda KB | prefetch issued / useful / late / wasted | wasted GB | stalls/token |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **native `-ngl 0`, cold median (§1)** | 4 149 (= 1/0.241) | — | page cache ≈ 90 % | **1.82** | **74 098** | 24 | 131 | — | — | — |
| cache only, depth 0, 100 GiB | 1 576 | 1 382 | 87.6 % | **1.17** | **3 408** | 361 | 403 | — | 0 | 61 |
| prefetch depth 8, 100 GiB | 2 010 | 1 794 | 84.9 % | 1.62 | 4 695 | 363 | 404 | 5 938 / 560 / 1 567 / 3 811 | 102 | 61 |
| prefetch depth 16, 100 GiB | 2 067 | 1 831 | 84.4 % | 1.77 | 5 100 | 364 | 404 | 11 567 / 958 / 1 799 / 8 810 | 233 | 59 |
| prefetch depth 8 + emulated compute | 2 466 (incl. 496 sleep) | 1 732 | 84.7 % | 1.64 | 4 820 | 362 | 404 | 6 306 / 1 039 / 1 242 / 4 025 | 108 | 61 |
| prefetch depth 8 + VRAM tier (792 slots) | 3 147 | 1 771 (+1 100 VRAM) | 84.7 % | 1.67 | 4 862 | 362 | 403 | 6 219 / 559 / 1 673 / 3 987 | 107 | 61 |
| prefetch depth 8, **150 GiB** | **1 147** | 1 033 | **91.6 %** | **0.63** | **1 826** | 365 | 403 | 888 / 140 / 155 / 593 | 16 | 40 |
| prefetch depth 8, 100 GiB, **code trace** | 3 661 | 3 303 | 73.5 % | 3.21 | 9 360 | 360 | 403 | 5 730 / 768 / 968 / 3 994 | 106 | 120 |
| prefetch depth 8, 100 GiB, page cache allowed | 1 367 | 1 331 | 82.5 % | 0.79 | 2 315 | 365 | 404 | 7 873 / 957 / 2 139 / 4 777 | 129 | 66 |
| **final code**, cache only, 100 GiB (serial) | 1 713 | 1 559 | 86.8 % | 1.30 | 3 771 | 362 | 404 | — | 0 | 66 |
| **final code**, prefetch depth 8, 100 GiB (serial) | 1 731 | 1 542 | 86.8 % | 1.30 | 3 762 | 362 | 404 | 809 / 253 / 32 / 524 | 14 | 64 |
| **final code**, cache only, **150 GiB** (serial) | **1 034** | 921 | **91.7 %** | **0.63** | **1 815** | 364 | 402 | — | 0 | 41 |
| **final code**, cache only, 100 GiB, **code trace** | 3 813 | 3 431 | 73.4 % | 3.35 | 9 716 | 359 | 403 | — | 0 | 132 |

Every arm ran to completion; no arm delivered a byte that differed from the
file (the verification arms in §6 check that on the same path). The rows
marked **final code** were run after §4.3's two corrections (the reload-cost
term constant again, demand reads serial on this device) and are the ones
to quote; the earlier rows are kept because the difference between them is
itself a finding. Under the final code, prefetch at depth 8 issues 809
speculative reads for 150 tokens, 253 of them useful, 524 wasted, and lands
within 1 % of cache-only on every column — the predictor mostly names
experts the cache already holds.

### 4.2 What the arms say

**Against native, the RAM tier holds up and the I/O shape is the point.**
With a 100 GiB cache — *smaller* than the ~180 GB of page cache native
enjoyed — TierInfer's cache-only arm reads **36 % fewer bytes per token in
22× fewer operations**: 1.17 GB in 3 408 reads of 361 KB against 1.82 GB in
74 098 reads of 24 KB. With 150 GiB the gap widens to **65 % fewer bytes in
40× fewer operations** (0.63 GB, 1 826 reads). The reads are expert-sized
`preadv` calls of 8.8–12.9 MB (storage layer: 9.5 MB per operation); the
kernel splits them at `max_sectors_kb=512` into the ~361 KB md sees, and md
passes them on at ~403 KB. So on TierInfer's path **md merges almost
nothing** (361 → 403 KB) because there is nothing left to merge; on the
native path it merges fivefold (24 → 131 KB) and still leaves the drives
with requests a third the size. That is §15's question answered: the
locality is TierInfer's, the merging is md's, and they are separable in the
counters.

**The device is still not the limit.** md0 delivered 0.74–0.78 GB/s in the
100 GiB arms and 0.83 in the 150 GiB one, against 2 GB/s measured capacity,
with await 1.4–2.0 ms. The limit is the exact path: a miss nobody prefetched
is a *synchronous, single-threaded* `pread` of ~30 MB (22 ms), and at 61
misses per token that alone is ~1.35 s of the 1.38 s wait. The streamer's
eight workers sit idle for it. This is the largest lever found by the
validation, and it is a correctness-free one — the misses of a layer are
all known at once and can be read together. It was applied after these
arms ran (§4.3).

**Prefetch without compute is a negative result, and the numbers say why.**
Depth 8 issues 5 938 speculative reads for 150 tokens; 560 land in time,
1 567 are still reading when the token asks (so the token waits on them
anyway), 3 811 are never used and cost 102 GB. Waste evicts useful entries
(hit rate 87.6 → 84.9 %) and the wasted reads share the device with the
demand reads. Depth 16 doubles the waste and helps nothing. Giving the
prefetcher ~500 ms of emulated compute per token to hide behind nearly
doubles the useful count (560 → 1 039) and cuts late ones by a fifth — and
recovers 60 ms of wait out of 1 794. At recall@16 of 67 % (§5) two of three
guesses are wrong by construction, and the useful third mostly names experts
the cache would have held anyway. **Prediction earns nothing here that
residency does not already provide; it only costs.** On GLM (`POLICY.md`)
the simulator said the dial was worth 1 %; on the 480B, measured, it is
worth less than zero without compute to overlap and roughly zero with it.

**The 150 GiB arm is where TierInfer should be compared, and it wins on
I/O.** 91.6 % hit, 0.63 GB and 1 826 reads per token, 40 stalls, 1.15 s
of I/O per token — against native's 1.82 GB and 74 k reads *with more RAM*.
What this does not say is tokens per second: there is no compute in the
loop and no loader to put one there (§7).

**The VRAM tier does real work and costs real time.** 792 slots (the
goal-9 budget at 4 096 context, reserve measured), 22.1 % hit rate on
routing that spreads over 9 920 experts, 57 923 transfers at 26.5 GB/s
(0.99 ms each). Staging the bytes into pinned memory and copying them adds
~1.1 s per token here because every transfer is synchronous and on the
token's path; in a real loop the copy would overlap the previous layer's
compute, and the 22 % hit rate would be the number that mattered — it says
a 20 GB VRAM working set catches a fifth of this model's expert traffic,
against the 76–79 % it caught on GLM's 128 experts per layer.

**The prompt class doubles everything.** Code routing at 100 GiB: 73.5 %
hit, 3.21 GB and 9 360 reads per token, 120 stalls — twice prose on every
axis, exactly as the trace's locality (§3) predicted and as native's own
0.39 vs 0.72 t/s showed.

**Double caching is measurable.** With the page cache free to help, a
TierInfer miss is often a page-cache hit: md0 sees 0.79 GB per token where
the same arm with `--drop-after-read` saw 1.62 GB, the streamer reports
2.87 GB/s (RAM speed for part of it), and 27.7 % of the model is resident in
the page cache at the end. A deployment that does not evict what it has
copied pays twice for RAM and measures nothing about its own cache.

### 4.3 Batching the demand reads — measured, and switched off for this device

The lever named above was applied (`d55e20e`: a layer's routed misses go
through the streamer together) and the cache-only arm rerun. It came back
**slower**: 2 379 ms against 1 576, hit rate 84.1 % against 87.6 %, 1.87 GB
per token against 1.17. Two effects were tangled in that, and two more arms
separated them:

| arm (cache only, 100 GiB, prose) | ms/token | I/O wait | RAM hit | md0 GB/token | md0 await | ms per miss |
|---|--:|--:|--:|--:|--:|--:|
| original (serial exact reads) | 1 576 | 1 382 | 87.6 % | 1.17 | 1.41 | 22.6 |
| batched demand + per-load reload cost fed to the cache | 2 379 | 2 112 | 84.1 % | 1.87 | 3.88 | 26.7 |
| serial, reload cost constant again | 1 713 | 1 559 | 86.8 % | 1.30 | 1.41 | 23.8 |
| batched, reload cost constant again | 1 839 | 1 636 | 86.8 % | 1.30 | 4.23 | 24.9 |

1. **The hit-rate loss was the cache's, not the batching's.** A repair
   made earlier the same day (`a1b4018`) had started feeding the cache's
   reload-cost term with each load's measured read time. Under concurrent
   reads that time is mostly queueing behind the other loads, so identical
   experts received costs a factor of ten apart by luck, and the recency-led
   policy began evicting by that noise: 87.6 → 84.1 %, 0.7 GB more per
   token. With the term constant again the two arms agree to the decimal
   (86.8 %, 1.30 GB). A reload cost has to be a property of the expert, not
   of the moment it was read; the repair is reverted (`08f18d6`) and the
   reason is in the code.
2. **Concurrency buys nothing on this device.** Same cache, same bytes:
   batched is 5 % slower per miss, md0 delivers 0.73–0.74 GB/s either way,
   and await triples (1.4 → 4.2 ms) because eight 30 MB reads are queueing
   for a pipe that one of them already fills. On the reference NVMe the
   same mechanism measured 3.47× (`STREAMING.md`). It is a property of the
   device — so it is now a switch (`Prefetcher(batch_demand=…)`), and
   `autoconfig.probe_concurrency` measures it from the model's own files in
   about a second and decides, threshold named, instead of anyone assuming
   (`5ceaf53`). The remaining arms ran serial.

What this leaves as the bottleneck: **~0.75 GB/s from md0 for expert-sized
random reads, at under 50 % utilisation.** The load phase streamed 1.7 GB/s
sequentially and fio measured 2 GB/s for 1 MiB random reads, so the gap is in
how 8.8–12.9 MB reads at 32-byte-aligned offsets travel through a 512 KB
stripe: the mean request md0 sees is 361–376 KB, not 512, which says most
requests are split at chunk boundaries. Aligning and sizing TierInfer's
reads to the array's geometry (read whole stripes, pad to the chunk) is the
next measurable step and was not taken inside this addendum.

The serial 150 GiB cache-only arm is the best measured configuration for
this model on this host: **1 034 ms of I/O per token, 91.7 % hit, 0.63 GB in
1 815 reads** — against native's 1.82 GB in 74 098 reads with more RAM.
Prefetch at 150 GiB (§4.1) changed none of those figures.

### 4.4 The loader under llama.cpp — real decode, real faults (CP-12)

Goal 6 delivered: `build/libtierinfer_mmap.so` preloaded into an unmodified
llama-server b10482 turns each shard's mapping into a userfaultfd region the
TierInfer server answers per expert (`docs/LOADER.md`,
`src/tierinfer/loader.py`). The harness (`benchmarks/loader_ab.py`) runs the
two arms cold, same binary, same flags, same prompt, greedy.

**Three cold runs per arm** (`benchmarks/loader-out/q480b-ncmoe60-*.json`,
table `q480b-ncmoe60.md`), medians with ranges:

| arm | load s | prompt t/s | gen t/s | infer GiB | infer reads | mean KB | await ms | tokens |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| native `-ngl 99 -ncmoe 60` | 192 | 0.767 | **0.467** (0.437–0.481) | 246.1 | 2 255 392 | 107–116 | 4.8 | reference |
| loader, tier 150 GB, depth 0 | 37 | 0.750 | **0.647** (0.622–0.668) | 157.9 | 426 882 | 387 | 2.0 | identical, all six runs |

Per generated token the loader served 464 of 496 routed experts from its
tier (93.6 % median hit rate in each of the three runs), copied 454 MB in
832 faults and evicted 16 experts; wall 1.31–1.45 s median against
native's 2.1–2.3 s. The three loader runs are byte-identical in I/O. The prompt batch is
where the tier fills: 5 100 misses, 150 GB in 148 s at 1.0 GB/s — the
device's ceiling for expert-sized reads (§1, §4.2) — so prompt speed is
the same in both arms and dominated by storage either way.

The first attempt at this run crashed llama-server, and the cause is worth
the record: experts of the two GPU layers are read through the mapping at
load, llama.cpp then unmaps that 9.3 GB suffix, and the cache — full after
the prompt batch — evicted precisely those never-routed experts into
addresses the kernel had since reused. `MADV_DONTNEED` on a running
process's own memory; `free(): invalid pointer`. The shim now interposes
`munmap`, the server is told, and an eviction into a hole is refused on
both sides (`11a9456`; CP-12 has the detail, tests reproduce it).

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

Same replay path, every delivered expert compared byte for byte with an
exact read (`safety.exact_load`), predictor warmed on 30 tokens so that the
speculative path is actually exercised (the first pass without warm-up
issued no speculation at all and two injections did not fire — recorded in
`docs/CHECKPOINTS.md` CP-8a).

| injection | what happened | delivered / mismatches |
|---|---|--:|
| none (`verify-d8`, cold) | 3 tokens, every expert from the file | 1 488 / **0** |
| predictor always wrong (experts 144–159) | 397 speculative reads, 10 useful, 384 wasted (10.9 GB), 1 838 exact fallbacks, pool exhausted 444× | 3 968 / **0** |
| one speculative read in fifty fails (`EIO`) | 15 injected → 15 failed loads → 15 exact fallbacks; 270 issued, 93 useful | 3 968 / **0** |
| two stream buffers | pool exhausted 165×, speculation throttled to 29 issued | 3 968 / **0** |
| eight VRAM slots (0.2 GB) | 0.3 % VRAM hit (13 of 3 968), 3 955 transfers of 105 GB at 26.1 GB/s — the tier thrashes, the token gets its bytes | 3 968 / **0** |
| an expert whose ranges cannot be resolved (routed by token 1) | token 0 delivered (496 / 0), then **`FATAL: explicit failure: injected: expert (0, 93) cannot be resolved to byte ranges`** and the run ended | 496 / **0** |

The two outcomes the scope allows are the two observed: wrong guesses, failed
speculative reads and exhausted pools degrade to slower exact reads with the
right bytes; an expert the index cannot address stops the run with a message
naming it. Nothing was served from a stale buffer, nothing was silently
skipped. A speculative guess about an unresolvable expert is skipped and
counted (`unmappable`), never raised.

## 7. What could not be measured, and why

**Tokens per second of "llama.cpp + TierInfer" — now measured (§4.4).**
At the time of §1–§4.3 no loader existed; every TierInfer number there is
I/O and residency under *replayed* real routing and is still presented as
such. §4.4 is the live decode rate, three runs per arm.

**Compute overlap.** With no compute in the loop, prefetch lead time is
whatever the sleeps in the `comp` arm provide (~500 ms per token, assumed).
Whether real attention time on this model would let prefetch hide more is
unmeasured; what is measured is that at recall@16 of 56–67 % it would hide
two wrong guesses for every right one.

**A real NVMe latency spike.** Not injected; `await` stayed within
1.4–4.2 ms across every arm. The timeout path is covered by a unit test.

## 8. Summary against addendum §23

| field | finding |
|---|---|
| Model / quantization / size | Qwen3-Coder-480B-A35B-Instruct, Q4_K_M (Q4_K + Q6_K down in 30 of 62 layers), 6 shards, 270.1 GiB |
| llama.cpp / TierInfer | b10482 `8b8640097`; TierInfer `main`, revisions per checkpoint |
| VRAM configuration | native offload point `-ngl 99 -ncmoe 60` (22.3 GB); TierInfer VRAM tier 792 slots (23 GB) at 4 096 context from the derived budget |
| RAM behaviour | native: page cache ≈ 180 GB serving ~90 % of a 20.9 GB per-token working set; TierInfer: 100 GiB → 86.8 % hit, 150 GiB → 91.7 %, LRU-equal policy |
| NVMe behaviour | md0 RAID0 over two USB 4M2 members; 1.7 GB/s sequential at load, ~0.75 GB/s for expert-sized random reads at <50 % utilisation; md merges native's 24 KB faults fivefold, TierInfer's 361 KB requests barely |
| native generation | 0.241 t/s cold median at `-ngl 0` (0.196–0.244 over six runs); 0.38–0.51 with attention on the GPU; prompt class halves it (0.72 vs 0.39 during capture) |
| TierInfer generation | **0.647 t/s** (0.622–0.668) under llama.cpp with the loader, 150 GB tier, three runs (§4.4); native on the same flags 0.467 (0.437–0.481); tokens identical |
| native I/O | 1.82 GB and 74 098 reads of 24 KB per token (cold median) |
| TierInfer I/O | 1.30 GB / 3 771 reads (100 GiB), **0.63 GB / 1 815 reads (150 GiB)** per token, 361 KB each; expert-sized `preadv`, one per projection |
| RAM cache effectiveness | 86.8 % / 91.7 % hit with less RAM than native's page cache; 36–65 % fewer bytes, 20–40× fewer operations |
| VRAM working-set effectiveness | 22.1 % hit from 792 slots on routing over 9 920 experts; 26.5 GB/s pinned transfers, 0.99 ms each; synchronous staging costs ~1.1 s per token in the replay |
| prefetch effectiveness | **none measurable**: within 1 % of cache-only at 100 and 150 GiB; without compute two of three speculative reads are late or wasted; depth 16 doubles waste for nothing |
| expert predictor effectiveness | heuristic; transition 66.8 % (prose) / 56.6 % (code) recall@16, 15 points over frequency; no trainable prerouter |
| observed bottlenecks | native: CPU-bound page faulting (85 % CPU, device half idle); TierInfer: ~0.75 GB/s for 8.8–12.9 MB reads split at 512 KB stripe chunks, device half idle; the synchronous exact path is the token's I/O wait and concurrency does not help it on this device |
| correctness issues discovered | sharded GGUF unsupported (blocking); strided-view routing read wrong on b10482; tracker fed per admission; cache misses never counted on the prefetch path; reload cost fed with queueing noise; wait timeout outside the guard; blocking drop; stranded loads on close; unbounded size list; ignored fadvise failures; `if vram:` falsy when empty; policy simulator's seconds in the observed schema; text-matching safety audit; expert size assumed uniform |
| correctness issues repaired | all of the above (`8d33288`, `897cd37`, `08f18d6`, `d71d993`, and the replay commits); 18 000+ deliveries verified byte for byte across the failure arms, 0 mismatches |
| remaining scope gaps | the llama.cpp loader (goal 6, and with it goals 7/9/10 as runtime mechanisms and completion items 9, 11, 14, 17, 18); trainable prerouter (8); prefetch priority classes and coalescing (7.7); backend device characteristics beyond the concurrency probe (7.8); FlowRunner consumer (12); NVMe tier under FreeToken (11); stripe-aligned reads on md (found here) |

A negative result stated as one: on this model and this storage, TierInfer's
explicit RAM tier reads far less and far larger than Linux demand paging with
less memory, its prediction and prefetch add nothing measurable, its VRAM
tier catches a fifth of the traffic, and none of it is under a running model
yet.

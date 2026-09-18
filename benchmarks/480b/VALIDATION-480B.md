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

_(table filled at CP-4; cold1/cold2/warm1 so far)_

| run | load s | prompt t/s | gen t/s | gen: md0 GB/token | md0 reads/token | md0 mean KB | sda mean KB | await ms | md0 util | CPU busy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cold1 | 191 | 0.745 | 0.241 | 1.82 | 73 142 | 25 | 134 | 4.43 | 0.46 | 0.84 |
| cold2 | 183 | 0.833 | 0.220 | 1.77 | 74 098 | 23 | 125 | 4.24 | 0.43 | 0.85 |
| warm1 | 0 (server kept) | 2.084 | 0.225 | 1.36 | 63 985 | 22 | 98 | 3.19 | 0.39 | 0.89 |

The first thing the table says: native generation is **not bandwidth-bound**.
md0 is under half utilised, delivering ~0.4 GB/s of a device measured at
2 GB/s, while the CPU is 85 % busy — busy faulting. Each token issues ~70 000
reads of ~23 KB (logical) which md merges into ~125 KB physical requests
before the drives see them: the merging is real, it is a factor of five, and
it is Linux's and md's, not anyone else's. The page cache is doing most of
the tier work already: a token needs 20.9 GB and native reads 1.8 GB of it,
so ~90 % of what a token touches was still resident from recent tokens.

## 2. GPU offload point (`-ngl 99 -ncmoe N`)

_(CP-5, running)_

## 3. Routing captured from the running 480B

_(CP-6: two 400-token traces, prose and code prompts; expert identities,
activation frequency, hot/cold, horizon)_

## 4. TierInfer replay — real routing, real files, real caches, no compute

_(CP-6/7/8: cache-only, prefetch depth 8/16, emulated compute, VRAM tier,
150 GB RAM tier, page cache allowed; per arm: cache hit rate, prefetch
useful/late/wasted/stalls, md0 GB and reads per token against native, sda
request sizes, VRAM hit rate and transfer rate)_

## 5. Predictor on 480B routing

_(CP-7: recall@k of frequency / persistence / transition / adaptive, per
prompt class; there is no trainable prerouter — see the audit)_

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

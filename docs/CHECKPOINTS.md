# Checkpoints — SCOPE Addendum 480B

Durable state for `docs/SCOPE-ADDENDUM-480B.md` §21. Newest at the bottom.
Each entry: revision, what was tested, exact command where relevant, result,
measured evidence, unresolved problems, next action. The work must be
resumable from this file and the repository alone.

## Host facts (measured 2026-09-18, after reboot at 02:44 CEST)

| item | value |
|---|---|
| kernel | 7.0.0-31-generic |
| CPU | 32 threads |
| RAM | 187 GiB total; 189 GB `MemAvailable` once FreeToken was stopped |
| GPU | RTX 5090, 32 607 MiB; driver 580.173.02; CUDA 13.0.3 |
| swap | /swap.img 8 G + /swap2.img 32 G, both 0 B used |
| cgroup | `memory` delegated to the user slice (ceilings via `systemd-run --user --scope`) |
| storage | `/dev/md0` RAID0 of `sda`+`sdb` (OWC Express 4M2, USB), 512 k chunk, ext4 `stripe=512`, mounted `/data/ai-data`; `max_sectors_kb=512`, `read_ahead_kb=4096`, members `mq-deadline` |
| model | `/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/*.gguf`, 6 shards, 290 058 826 208 bytes total (270.2 GiB) |
| llama.cpp | `~/llama.cpp-qwen38/build/bin/llama-cli` = build 10482 (8b8640097, 2026-08-18); has `cb_eval`, `ffn_moe_topk`, `-ncmoe`, `-ot`. `~/llama.cpp` = b9888 (what `build/tierinfer-trace` currently links) |
| previous boot | ended 02:43:22 with `nvidia-modeset: Error while waiting for GPU progress` repeating from 02:42:54; the GPU had stopped answering at ~22:35 the evening before (`nvidia-smi: Unable to determine the device handle`). Cause not established; TierInfer's VRAM tests and Ollama were both running that evening. |
| FreeToken | `freetoken-qwen38-flash-next-abliterated.service` (system slice, port 8092) held 25.6 GB VRAM and ~63 GB shmem at boot; the Human stopped it at ~03:10 for this work. |

## CP-1 — quality audit complete

- **Revision:** `6cc3f68` audited; this checkpoint committed on top.
- **Tested:** every module read end to end; `pytest -q` → 318 passed;
  `PYTHONPATH=src python3 tools/smoketest.py` → 10 PASS, 4 FAIL while
  FreeToken held the card (budget correctly negative), all PASS expected
  with the card free (re-run recorded at CP-2).
- **Result:** `docs/AUDIT-2026-09-18.md`. Headline: real, measured components;
  no production inference path uses any of them. One blocking defect for the
  480B (no sharded-GGUF support). Goals 6, 10 un-marked as complete; 8, 11,
  12 marked partial/disconnected.
- **Evidence:** file:line citations in the audit; external checks against
  `~/freetoken-qwen38` (flags exist), `~/FlowRunner` (no consumer),
  `~/llama.cpp-qwen38` (API present), the six shards (metadata read through
  `tierinfer.gguf`: `qwen3moe`, 62 layers, 160 experts, 8 used,
  `split.count=6`, types Q4_K/Q6_K/F32, every shard's last tensor ends at EOF).
- **Unresolved:** everything under "What this audit does not repair".
- **Next:** CP-2 correctness repairs, starting with sharded GGUF.

## CP-2 — correctness repairs complete

- **Revision:** the commit carrying this entry (on top of `96ffe4e`).
- **Repaired (audit items in parentheses):**
  1. Sharded GGUF (1, 2, 3): `gguf.read_model` / `shard_paths` read a
     `gguf-split` set as one model and check `split.no`, `split.count`,
     `split.tensors.count`; `TensorEntry.path` and `ByteRange.path` name the
     shard; `StorageBackend` holds one descriptor per file (`fd_for`,
     `for_model`), `ExpertStreamer` reads through it; `bench` measures
     residency, cold cache and warm-up over every shard and samples md
     member devices beside `md0`; `cli`, `smoketest`, `pin_floor` follow.
     `tests/test_shards.py` builds a three-shard split and reads across the
     seams.
  2. Prefetcher (6, 8): the tracker is told about a token once, at
     `end_token`, with everything it routed to; a prefetch still reading
     when demand names it is counted `late` with its wait, apart from
     `useful`; a `wait` timeout goes to the exact path and the load is
     orphaned and reaped, never leaked; `drop_unused` never waits.
  3. Streamer (3): `close()` fails anything still queued instead of
     stranding its waiter.
  4. Storage (2): operation sizes are a bounded histogram; a refused
     `posix_fadvise` raises instead of reporting an eviction that did not
     happen; a short read is checked before it is counted.
  5. Policy / telemetry (9, 10): `tierinfer.policy` states in its first
     paragraph that it is a simulation; its counters export under `sim.`,
     not beside observed ones.
  6. Safety audit (15): checks names bound or called (AST), not text.
  7. Trace tool (12): reads the routing tensor **row by row through
     `nb[1]`** — from b10482 `ffn_moe_topk` is a strided view of the
     argsort, and the old contiguous read returned wrong experts for every
     prompt token after the first; the assist mode and its flags,
     `tools/trace/expert_map.py`, `tests/test_expert_map.py` and
     `benchmarks/inloop.py` are removed (the measurement that falsified
     them stays in `REAL-ROUTING.md` / `inloop-report.json`); `--cpu-moe`
     (tensor-buffer override, same as `llama-cli -cmoe`) and `--one-by-one`
     added; timings printed; built against b10482 by default.
  8. Docs (17): README status, `docs/architecture.md`, `benchmarks/README.md`
     and `SCOPE.md` goals 2, 6, 8, 10, 11, 12 reconciled to the audit.
- **Tested:** `pytest -q` → 325 passed (329 before the four expert-map
  tests were removed with their subject, plus `test_shards.py`);
  `PYTHONPATH=src python3 tools/smoketest.py` → 14 passed, 0 skipped,
  0 failed with the card free (goal 9: 64 slots at 26.2 GB/s; goal 14:
  2 498 VRAM + 10 765 RAM experts on GLM).
- **Not repaired, by decision (audit "What this audit does not repair"):**
  the loader; trainable prerouter; priority classes; prefetch-path
  coalescing; backend device characteristics; FlowRunner consumer; NVMe tier
  under FreeToken.
- **Next:** CP-3, validate the 480B GGUF (already partly done at CP-1 via
  `tierinfer.gguf`; llama.cpp side running).

## CP-3 — 480B GGUF validated

- **Revision:** the commit carrying this entry.
- **Tested (addendum §6, §7):**
  1. Six shards present in
     `/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/`;
     sizes 49 973 553 312 / 49 768 927 168 / 48 862 488 352 / 48 952 192 064 /
     49 721 298 176 / 42 780 367 136 — byte-identical to the addendum's list;
     total 290 058 826 208 B = 270.1 GiB.
  2. `tierinfer.gguf.read_model` on shard 1 (post-CP-2): GGUF v3,
     `general.architecture=qwen3moe`, `general.file_type=15` (Q4_K_M),
     `split.count=6`, `split.no` 0..5 in order, `split.tensors.count=747`
     = tensors found (127+135+124+129+132+100); tensor types Q4_K / Q6_K /
     F32 only; **every shard's last tensor ends exactly at its file's EOF**
     (the smoketest goal-2 check, now per shard). Model: 62 layers, all MoE,
     160 experts, 8 used per token, no shared expert, `head_count_kv=8`,
     `key_length=value_length=128`, context 262 144. One expert =
     8 847 360 + 8 847 360 (gate/up, Q4_K) + 12 902 400 (down, Q6_K) =
     30.6 MB; 496 experts per token = 15.2 GB; routed total 303.4 GB of…
     see `tierinfer inspect` in CP-4 for the floor/routed split.
  3. llama.cpp b10482 (`~/llama.cpp-qwen38/build/bin/llama-cli`),
     conservative first run, exact command:
     `llama-cli -m <shard 1> -ngl 0 -c 512 -n 8 -t 32 -st --temp 0 -p "The capital of Denmark is"`
     → exit 0, output `The capital of Denmark is Copenhagen.`, wall 5 m 12 s,
     RSS 183.5 GB (mapped pages), 229 873 major faults; md0 read **432 GB**
     for the 290 GB model (8 972 229 reads, 47 KB mean logical; sda/sdb
     each ~985 k reads at 213 KB mean physical). That all six shards were
     resolved follows from the bytes read exceeding any subset of shards
     and from the coherent output; llama.cpp's own `print_info` lines are
     not emitted by this build's `llama-cli`/`llama-server` logs and will be
     taken from `tierinfer-trace`'s stderr (which carries `llama_log`) in
     CP-6.
  4. `llama-server` (same build) loads the model from shard 1 and answers
     `/completion` coherently in every baseline run so far (CP-4).
- **Result:** the model and the runtime are compatible; no llama.cpp change
  was needed.
- **Unresolved:** none for this checkpoint.
- **Next:** CP-4 native baseline (running: `native-ngl0-cold{1,2,3}`,
  `warm{1,2,3}`).

### Incident during CP-4 runs, recorded here because it changed the method

The first runner started `llama-server` under `/usr/bin/time -v` and later
killed `$!` — which was `time`, not the server. The server survived as an
orphan on port 8931; the next run's server failed to bind, the runner's
health poll succeeded against the orphan, and "warm1" ran against cold1's
still-mapped server (valid as a warm run, and kept), while the "cold2" that
followed had 54 % of the model still resident behind a live mapping — the
project's own goal-6 finding, met again. cold2 was discarded and rerun
after the orphan was stopped; the runner now owns the server's pid, refuses
a busy port, and reads `VmHWM`/`majflt` from `/proc` before stopping it.
Raw artefacts of every run, including the discarded one's log lines, are in
`benchmarks/480b/raw/`.

## CP-4 — native llama.cpp baseline complete (`-ngl 0`)

- **Revision:** the commit carrying this entry.
- **Command (each run):**
  `serve_run.sh <label> <out> 32 -- -m <shard 1> -ngl 0 -c 4096 -t 32 --no-warmup -fa off`
  = `llama-server` b10482 on 127.0.0.1:8931, wait for `/health`, one
  `/completion` with the 111-token prompt in `raw/prompt.txt`, `n_predict 32`,
  `temperature 0`, `cache_prompt false`; `/proc/diskstats` for md0/sda/sdb
  sampled every 2 s; cold = `tierinfer.bench.drop_cache` over the six
  shards (residency 0.0 % / 0.0008 % after).
- **Result:** six runs, three cold and three warm, table in
  `benchmarks/480b/VALIDATION-480B.md` §1 and `raw/base/summary.json`.
  Generation **0.241 t/s cold median (0.220–0.244), 0.201 warm median
  (0.196–0.225)**; prompt 0.745 / 1.079 t/s; load 171–201 s (MAP_POPULATE).
  Per generated token: **1.82 GB cold / 2.18 GB warm from md0 in ~74 000 /
  84 000 reads of 22–26 KB (logical), merged by md to 98–147 KB at the
  members**; await 3.2–4.8 ms; md0 utilisation 0.39–0.48; CPU 83–89 % busy.
- **What it says:** native generation is CPU-bound faulting, not
  bandwidth-bound — the device runs under half utilised at ~0.4 GB/s. The
  page cache already serves ~90 % of a token's 20.9 GB working set. md's
  merging is a factor of ~5 and belongs to md.
- **Trace tool validated on the new build:** GLM-4.5-Air, the same 30-token
  prompt decoded batched and one token per `llama_decode`: routed-expert
  agreement **97.9 % (10 348/10 568), no token below 95 %**; consecutive
  prompt rows overlap by 9.7 %, so the rows are distinct tokens, not copies.
  The strided read is right; the residual 2 % is batched-vs-single numerical
  difference at marginal experts. (`raw/glm-batched.jsonl`,
  `raw/glm-onebyone.jsonl`.)
- **Unresolved:** RSS/faults missing for cold1 and warm1 (runner defect,
  fixed from cold2).
- **Next:** CP-6 routing capture from the 480B is running (`chain3`), CP-5
  GPU sweep queued behind it (`chain4`), replay arms behind that (`chain5`).

### CP-3 addendum — llama.cpp's own recognition of the model

From `tierinfer-trace`'s stderr (`raw/trace-prose.err`), which carries
`llama_log`: `llama_model_loader: additional 5 GGUFs metadata loaded.`,
`print_info: file type = Q4_K - Medium`, `file size = 270.13 GiB (4.83 BPW)`,
`arch = qwen3moe`, `n_layer = 62`, `n_expert = 160`, `n_expert_used = 8`.
Addendum §6 items 3–6 are therefore shown by the runtime itself, not only
inferred from bytes read.

## CP-6a — expert activity observed from the running 480B (two traces)

- **Revision:** the commit carrying this entry.
- **Command:** `build/tierinfer-trace -m <shard 1> -ngl 99 --cpu-moe -t 32 -c 4096 -n 400 -f <prompt> -o traces/qwen3coder480b-<arm>-400.jsonl`
  (llama.cpp b10482 via `libllama`, `cb_eval` on `ffn_moe_topk-<layer>`,
  strided rows; `--cpu-moe` = the `-cmoe` buffer-type override).
- **Result:** `traces/qwen3coder480b-prose-400.jsonl` and `-code-400.jsonl`,
  24 862 routing lines each (62 layers × 401 decodes), read cleanly by the
  strict reader; reports in `benchmarks/480b/routing-{prose,code}.json`,
  table in `VALIDATION-480B.md` §3 and §5.
- **Measured:** capture ran at 0.72 t/s (prose) and 0.39 t/s (code) with
  attention on the GPU; a token uses all 496 layout experts (20.9 GB);
  prose: 72 % of experts seen in 400 tokens, top-10 % share 56 %, neighbour
  overlap 38.5 %, 128-token horizon 173 GB; code: 91 % seen, 40 %, 26 %,
  221 GB. Predictors (heuristic, no prerouter): transition 66.8 % / 56.6 %
  recall@16, frequency 51.9 % / 37.6 %.
- **Addendum §13 items:** expert identification and range mapping (index,
  per shard), activation observation (trace), frequency tracking and
  hot/cold (report) — done. Residency / prefetch / eviction *decisions* are
  measured in the replay arms (CP-6b/7/8).
- **Unresolved:** none.
- **Next:** CP-5 GPU sweep is running (`chain4`); replay arms queued.

## CP-5 — GPU offload baseline selected

- **Revision:** the commit carrying this entry.
- **Command:** `serve_run.sh <label> <out> 32 -- -m <shard 1> -ngl 99 -ncmoe N -c 4096 -t 32 --no-warmup -fa on`
  for N = 62, 60, 58, 57 (cold, then warm), stop at the first failure.
- **Result:** table in `VALIDATION-480B.md` §2, raw in `raw/sweep/`. VRAM
  after load 12.9 / 22.3 / 31.6 GB for 0 / 2 / 4 expert layers on the card;
  N=57 fails in `cudaMalloc` for a 29.8 GiB buffer (`raw/sweep/native-ngl99-ncmoe57-cold.server.log`).
  Generation 0.38–0.51 t/s across the working configurations against 0.20–0.24
  at `-ngl 0`; md0 reads per token 51–85 k.
- **Selected:** `-ngl 99 -ncmoe 60` (22.3 GB, 9 GB headroom). N=58 works with
  under 1 GB free and is not stable enough to build a comparison on.
- **Unresolved:** none.
- **Next:** CP-6b/7/8 replay arms are running (`chain5`), starting with the
  byte-verification arm.

## CP-8a — failure behaviour on the real model (first pass)

- **Revision:** the commit carrying this entry.
- **Command:** `benchmarks/replay.py <shard 1> traces/qwen3coder480b-prose-400.jsonl --depth 8 --predictor-warmup 30 --tokens 8 --verify-every 1 --drop-after-read --ram-gb 60 --inject <mode>`
  under `systemd-run --user --scope -p MemoryMax=140G`; every delivered
  expert compared byte for byte with `safety.exact_load`.
- **Results** (`benchmarks/replay-out/480b/*.log`, `*.summary.json`):
  - `verify-d8` (no injection, cold, 3 tokens): 1 488 deliveries, 0 mismatches.
  - `inj-bad-predictor` (always predicts experts 144–159): 397 speculative
    reads issued, 10 useful, 384 wasted (10.9 GB), 1 838 exact fallbacks
    counted as stalls, pool exhausted 444 times; **3 968 deliveries, 0
    mismatches**. A wrong predictor costs bandwidth and time, nothing else.
  - `inj2-fail-reads` (one speculative read in fifty raises `EIO`): 15
    injected → 15 failed streamer loads → 15 exact fallbacks; 270 issued,
    93 useful, 24 late; **3 968 deliveries, 0 mismatches**.
  - `inj2-tiny-pool` (two buffers): pool exhausted 165 times, 29 issued,
    12 useful; **0 mismatches**. Speculation is throttled, delivery is not.
  - `inj2-tiny-vram` (eight device slots): ran and delivered **0
    mismatches**, but its VRAM counters were lost to a bug in the harness
    (`if vram:` on an object whose `__len__` is 0 after `close()`); fixed,
    rerun queued as `inj3-tiny-vram`.
  - `inj-fail-reads`, `inj-tiny-pool` (first pass, no predictor warm-up)
    and `inj2-missing-range`: **the injection did not fire** — with eight
    tokens of history every prediction named an expert already resident,
    so no speculative read existed to fail, and the unresolvable expert
    (3, 7) was never routed to. Both are harness lessons, recorded: the
    injections were re-aimed (`--predictor-warmup`; the broken expert is now
    one the second replayed token routes to) and `inj3-missing-range` is
    queued.
- **Also found by this pass:** the cache reported a 100 % hit rate while
  every expert came from the file, because the prefetcher never told it
  about misses (fixed, `897cd37`). Ten defects in, the pattern the project
  keeps meeting: a counter that is only ever incremented on the happy path.
- **Unresolved:** `missing-range` and `tiny-vram` results pending rerun.
- **Next:** measurement arms running (`cache-d0-100g` first).

## CP-8b — failure behaviour complete

- **Revision:** the commit carrying this entry.
- **Result:** all six injections have fired and are in `VALIDATION-480B.md`
  §6. Re-aimed reruns: `inj3-missing-range` — expert (0, 93), routed by the
  second replayed token — ended the run at token 1 with
  `FATAL: explicit failure: injected: expert (0, 93) cannot be resolved to
  byte ranges` after 496 verified deliveries; `inj3-tiny-vram` — 8 slots —
  0.3 % VRAM hit, 3 955 transfers, 0 mismatches over 3 968 deliveries.
  Totals across the injection arms: **17 856 delivered experts compared
  with exact reads, 0 mismatches**; every degradation was slower, none was
  wrong; one failure was explicit and named its cause.
- **Addendum §18 coverage:** VRAM exhausted (tiny-vram), RAM pressure
  (100 GiB tier under a 140 G cgroup, 5–7 k evictions per arm), prefetch
  falling behind (`late` counted in every arm), a requested expert not
  cached (stalls, every arm), prediction wrong (bad-predictor), cache space
  exhausted (evictions; tiny-pool for the stream pool), an async read
  failing (fail-reads), a range that cannot be resolved (missing-range).
  NVMe latency spikes were not injected; `await` stayed 1.4–2.0 ms
  throughout and the timeout path is covered by a unit test
  (`test_a_wait_timeout_falls_back_to_the_exact_path`).
- **Next:** `chain6` — the same measurement arms with a layer's demand
  misses read concurrently (`d55e20e`), then CP-9 and the final
  reconciliation.

## CP-6b — TierInfer large-model run operational

- **Revision:** the commit carrying this entry.
- **What ran:** `benchmarks/replay.py` over the six shards on md0 with the
  captured 480B routing: `ExpertStreamer` (8 workers) + `ExpertCache`
  (100 / 150 GiB of real bytes) + `Prefetcher` (depth 0 / 8 / 16) +
  `VramResidency` (792 slots, real `cudaMemcpy`), page cache evicted after
  every read so TierInfer's cache is the RAM tier; telemetry JSONL per arm
  (`benchmarks/replay-out/480b/*.telemetry.jsonl`), md0/sda/sdb counters
  around and during. Eleven measurement arms of 150 tokens plus reruns.
- **Result:** table and reading in `VALIDATION-480B.md` §4.1–4.3.
- **What "operational" means here and does not:** the mechanisms move real
  bytes under real routing and every counter is observed; no model computes
  on them (no loader — audit item 12).

## CP-7 — expert, cache and prefetch telemetry validated

- **Revision:** the commit carrying this entry.
- **Expert telemetry:** identities from the router (`cb_eval`), ranges from
  the index per shard, activation and hot/cold from the trace report
  (CP-6a). **Cache:** hits, misses, evictions, bytes from the cache's own
  counters — after fixing the one that lied (misses were never counted on the
  prefetcher's path, `897cd37`) and the one that was fed noise (reload cost,
  `08f18d6`). **Prefetch:** issued / useful / late / wasted / stalls, with
  `late` split from `useful` by whether the load had finished before the
  routing asked (addendum §12's "useful prefetch means arrived before demand
  and consumed"). **VRAM:** hits, transfers, bytes, GB/s from
  `VramResidency`. **Device:** `/proc/diskstats` md0 + members, so
  TierInfer's requests, md's merged requests and the drives' physical
  requests are in one table (§15).
- **Three telemetry defects found by running it, all fixed:** a 100 % cache
  hit rate over a run that read everything from disk; a VRAM tier whose
  counters vanished because `if vram:` was false on an empty residency;
  `unmappable` missing from the summary.
- **Next:** last two chain7 arms, then CP-9 (comparison) and CP-10 (final
  reconciliation).

## CP-9 — native-vs-TierInfer comparison complete

- **Revision:** the commit carrying this entry.
- **Compared:** same model, same prompt class (prose; code as the second
  class), same storage, same host state; native = six `llama-server` runs
  at `-ngl 0` (three cold, three warm) and six across the offload sweep;
  TierInfer = replayed real routing through the real mechanisms, 150 tokens
  per arm, cold, page cache evicted after every read, three arms rerun
  after the corrections in §4.3. Medians, minima and maxima reported; no
  run selected.
- **Result** (`VALIDATION-480B.md` §4, §8): per generated token native reads
  1.82 GB in 74 098 requests of 24 KB (md merges to 131 KB); TierInfer reads
  1.30 GB in 3 771 requests at 100 GiB and **0.63 GB in 1 815 at 150 GiB**,
  361 KB each. RAM cache 86.8 % / 91.7 %. Prefetch within 1 % of cache-only.
  VRAM tier 22.1 % hit. Tokens per second of TierInfer: not measurable
  without a loader — stated, not estimated.
- **md merging accounted for (§15):** logical requests, TierInfer's issued
  I/O (9.5 MB per operation), md0's requests and the members' requests are
  all in the tables; the fivefold merge on the native path is md's.
- **Next:** CP-10.

## CP-10 — final reconciliation

- **Revision:** the commit carrying this entry (HEAD of `main`).
- **Repository vs durable state:** `SCOPE.md` goals 2, 4, 5, 6, 7, 8, 10, 11,
  12, 14 carry the audit's and the validation's words; `README.md`,
  `docs/architecture.md`, `benchmarks/README.md` agree with them;
  `docs/AUDIT-2026-09-18.md` is the audit, `benchmarks/480b/VALIDATION-480B.md`
  the deliverable, `benchmarks/480b/raw/` and `benchmarks/replay-out/480b/`
  the evidence. `pytest`: 338 passed. `tools/smoketest.py`: 14 passed on
  this host with the card free.
- **Addendum §24, item by item:** 1 audit — done (CP-1). 2 identified — done
  (audit table, 17 rows). 3 blocking defects repaired — done (shards;
  CP-2) plus the ones the runs found (CP-7, CP-8). 4 goal status reconciled
  — done (SCOPE.md; goals 6 and 10 un-marked). 5 shards validated — done
  (CP-3). 6 runtime loads it — done, no change needed (CP-3). 7 native
  baseline — done, six runs (CP-4). 8 offload configuration — done,
  `-ncmoe 60` (CP-5). 9 exercised under real pressure — done: 270 GiB model
  against 183 GB RAM and a 100/150 GiB tier under a cgroup, 23 GB VRAM tier.
  10 TierInfer movement distinguishable from mmap — done: page cache evicted
  after every read, md0/sda/sdb counted, requests of 361 KB against 24 KB.
  11 RAM cache from real accesses — done. 12 VRAM working set from real
  accesses — done (real `cudaMemcpy`, 22.1 %). 13 expert activity from real
  execution — done (two traces, `cb_eval`). 14 prefetch measured — done,
  negative. 15 predictor verified real/connected/measurable — done for the
  heuristic predictors, which are connected to the prefetcher; **no
  prerouter exists**, recorded. 16 repeated controlled comparison — done
  (n=3 native cold/warm; three replay reruns; medians/min/max). 17 md merging
  accounted for — done. 18 failure behaviour — done, six injections, 17 856
  verified deliveries. 19 recorded durably — this file, the validation, the
  raw directories, `git log`. 20 repository, tests, scope state and status
  agree — yes, as of this commit.
- **What remains open, in the scope's terms:** the llama.cpp loader (goal 6)
  and everything that becomes measurable only with it; the trainable
  prerouter (goal 8); prefetch priority classes and coalescing; device
  characteristics in the backend beyond the concurrency probe; the
  FlowRunner consumer; the NVMe tier under FreeToken; stripe-aligned reads
  on md (new, from this validation).

## CP-11 — the loader under a running llama.cpp (first live A/B, GLM-4.5-Air)

- **Revision:** `a1bea1e` (loader `f417eed` … `a8441dd`); this run used the
  code before `a8441dd`.
- **What ran:** `raw/../glm-first/ab.log`: llama-server b10482, `-ngl 0 -c
  4096 -t 32 --no-warmup`, 111-token prompt, 32 greedy tokens, cold page
  cache. Loader arm: `LD_PRELOAD=build/libtierinfer_mmap.so` +
  `tierinfer serve --ram-gb 27 --workers 8 --depth 0`. Native arm: no shim,
  `systemd-run --user --scope -p MemoryMax=32G`.
- **Result — correctness:** the 32 generated tokens are **byte-identical**
  between the arms (`glm-first/*.completion.json`). The loader's RSS held at
  29.1 GB (tier 27 GiB + floor slack) against native's 33.6 GB under its
  ceiling.
- **Result — performance (negative, first pass):** loader prompt 2.24 t/s,
  generation **0.086 t/s**; native under the ceiling 3.40 / **0.318 t/s**.
  Whole-run reads: loader 411 GB in 3.4 M requests of 126 KB; native 124 GB
  in 1.66 M of 78 KB.
- **Why, from the loader's own telemetry** (`glm-first/loader.telemetry.jsonl`,
  per generated token, median): 260 of 360 routed experts resident (72 %),
  100 misses (~1 GB), **3 193 faults and 8.9 GB copied**. llama.cpp's 32
  compute threads touch one expert's pages in parallel; every thread's
  fault arrives as an event, and a counter heuristic re-copied whole
  experts for a third of them — 36 000 repairs in 30 tokens. Fixed in
  `a8441dd`: the client's `/proc/<pid>/pagemap` says whether the page is
  present; present → wake, absent → copy. Also found and fixed on the way:
  the mapped-prefix/suffix llama.cpp unmaps (`3564d7c`, `1c8f1a6`), a lost
  wake-up on already-present pages (`bb883b3`), redundant re-copies of
  resident experts (`750b00e`).
- **Acceptance IDs moved:** TI-LLAMA-002 (activity during generation) and
  TI-LLAMA-010 (identical tokens) → VERIFIED; TI-LLAMA-006/007 (real storage
  reads, real RAM-tier hits) → VERIFIED; TI-LLAMA-011 → IN_PROGRESS (the
  harness runs with the fix are queued: GLM ×3, then 480B ×3).
- **Next:** CP-12 with the harness numbers.

## CP-12 — the loader under llama.cpp on the 480B (live A/B, three runs each)

- **Revision:** `39ba8ce` (loader `11a9456` + `e00bf01` + `39ba8ce`);
  harness `82d6a98`/`d021476`. Three runs per arm; the full table is
  `benchmarks/loader-out/q480b-ncmoe60.md` (`benchmarks/480b/harness_table.py`).
- **What ran:** `benchmarks/loader_ab.py`, llama-server b10482,
  `-ngl 99 -ncmoe 60 -fa on -c 4096 -t 32 --no-warmup`, the 111-token
  prompt, 32 greedy tokens, cold page cache before each run (`0.00 %`
  resident after drop). Loader arm: `LD_PRELOAD=build/libtierinfer_mmap.so`
  + `tierinfer serve --ram-gb 150 --workers 8 --depth 0`. Native arm: no
  shim, no ceiling (its page cache is the whole host, ~180 GB).
- **Incident first — the first attempt died.** `q480b-ncmoe60-loader-1-crash.*`:
  llama-server ended with `free(): invalid pointer` after 557
  `madvise(DONTNEED)` failures (ENOMEM). Cause, from the loader's own
  layout: `-ncmoe 60` puts the experts of layers 60–61 on the GPU;
  llama.cpp reads them through the mapping at load (served, 320 experts,
  9.3 GB) and then **unmaps that suffix fragment** of shard 6. The kernel
  handed those addresses to later allocations. When the prompt batch filled
  the 150 GB tier, the cache evicted its least valuable entries — exactly
  those 320 never-routed experts — and `MADV_DONTNEED` landed on llama's
  own memory. Fixed in `11a9456`: the shim interposes `munmap`, reports
  `UNMAP <addr> <len>`, and refuses any `EVICT` into a hole regardless;
  the server forgets what lived there, never prefetches it, and copies and
  evicts around holes. Two tests reproduce it through the preloaded shim.
  The rerun's server log confirms the diagnosis to the expert:
  `unmapped [32794058752, 42780364800) of …-00006-of-00006.gguf (9.30 GB);
  320 experts gone, 320 of them were resident`.
- **Result — correctness:** the 32 generated tokens are **identical**
  between the arms (`q480b-ncmoe60-{native,loader}-1.json`, `content`).
- **Result — performance, three runs each (median, range):**

  | arm | load s | prompt t/s | gen t/s | infer GiB | infer reads | mean KB | await ms |
  |---|--:|--:|--:|--:|--:|--:|--:|
  | native | 192 | 0.767 | **0.467** (0.437–0.481) | 246.1 | 2 255 392 | 107–116 | 4.8 |
  | loader (150 GB tier) | 37 | 0.750 | **0.647** (0.622–0.668) | 157.9 | 426 882 | 387 | 2.0 |

  Generation **39 % faster** than native (medians; the worst loader run beats
  the best native run by 29 %) at equal prompt speed, with 36 % fewer bytes
  and 5.3× fewer read requests from md0, each 3.4× larger. Load is 5× faster
  because nothing is populated up front. The loader's three runs are
  I/O-identical (157.9 GiB, 426–428 k reads; 93.6 % hit rate in each), so
  the spread in t/s (0.622–0.668) is compute-side noise, not the tier.
- **From the loader's telemetry** (`q480b-ncmoe60-loader-1.telemetry.jsonl`):
  the prompt batch routed 5 322 distinct experts, 5 100 misses, 150 GB
  copied in 148 s (1.0 GB/s, storage-bound — the same 0.75–1.0 GB/s ceiling
  §1 and §4 measured); the tier was full (6 053 entries, 150.0 GB) from
  then on. Per generated token, median: 464 hits / 32 misses (**93.6 %
  hit rate**, 87–96 %), 832 faults, 454 MB copied, 16 evictions, 1.45 s
  wall. Whole run: 59 212 faults, of which 47 502 were for pages already
  present (32 compute threads touching one expert: woken via
  `/proc/<pid>/pagemap`, not re-copied), 387 repaired; 0 repeat faults,
  0 EAGAIN, 0 read retries; 28 UNMAP messages, 5 851 experts forgotten
  (teardown included).
- **Acceptance IDs moved:** TI-LLAMA-004/005/009/011/012 → VERIFIED;
  TI-480B-009/010/011/012 → VERIFIED (loader); TI-PERF repeated-runs
  criterion met (three cold runs per arm, tokens identical in all six).
- **GLM harness (from CP-11's queue), for the record:** `glm-ngl0.md` —
  native 0.310 / 0.107 / 0.325 t/s, loader run 1 0.324 t/s with identical
  tokens; loader runs 2 and 3 **stood aside** (the shim found a stale socket
  file and ran native mmap: 0.01 GB of I/O, 57 GB RSS) and are not loader
  measurements. Fixed since (`stood_aside` flag, harness waits for accept,
  shim retries 10 s); the 480B runs above are with the fix.

## CP-13 — an NVMe tier under FreeToken (item 6; TI-FT-*)

- **Revision:** TierInfer `0966281` (`tierinfer.ftw`, `tierinfer.client`,
  `serve <ftw-dir>`, `ROUTED`, next-step prefetch); FreeToken checkout
  `~/freetoken-qwen38` on the local branch `tierinfer-tier` (`c50484b` on
  upstream `9535656`), the patch kept in `patches/freetoken-tierinfer-tier.patch`.
  Design: `docs/superpowers/specs/2026-09-18-freetoken-tier-design.md`;
  how it runs: `docs/FREETOKEN.md`.
- **What runs:** `benchmarks/freetoken_ab.py` on the Human's Flash-Next
  checkpoint (`qwen38-flash-next-abliterated-ftw-fixed`, 121 GB FTW, 48
  MoE layers × 512 experts, 2.77 MB per expert over six banks), `ft serve
  --moe-strategy offload --moe-cpu-layers 12` (layers 0, 4, …, 44 on the CPU
  executor; 36 on the GPU), 32 k context, greedy, 95-token prompt, 63
  generated tokens, page cache dropped before every run. Native: all banks
  read at load. Tiered: the 12 CPU-executor layers' banks (17.1 GB) are
  TierInfer-served buffers under a budget. Table:
  `benchmarks/freetoken-out/flashnext.md`.
- **Correctness:** greedy output identical across all eleven runs
  (native ×2, tiered 8 GB ×3, 16 GB ×2, 8 GB + prefetch ×2, smoke); the
  tiered banks' rows are the shards' bytes (TI-FT-010).
- **Result:**

  | arm | load s | wall s (95+63 tok) | FreeToken decode t/s | infer I/O | load I/O |
  |---|--:|--:|--:|--:|--:|
  | native (all 121 GB resident) | 55 | 4.3 | 34.8 / 35.0 | 0.01 GiB | 72.7 GiB |
  | tiered, 16 GB budget (holds the 12 layers) | 44 | 25.1 / 25.2 | **35.4 / 34.8** | 15.5 GiB | 57.3 GiB |
  | tiered, 8 GB budget (47 % of them) | 44 | 38.5–40.3 | 5.4–6.1 | 18.7 GiB / 83 k reads @ 235 KB | 57.3 GiB |
  | tiered, 8 GB + prefetch depth 8 | 44 | 38.2 | 6.3 | 18.7 GiB | 57.3 GiB |

  With the budget holding the layers, **decode is native speed** (0
  misses, 27 ms per step, FreeToken's own 35 t/s) and the whole cost is
  prefill: FreeToken's whole-layer pageable copy pulls every row through
  the tier (17.1 GB, 9 150 faults, ~23 s at 0.75 GB/s), which is the same
  bytes native reads at load — load 44 s + prefill 23 s against native's
  55 s + 0. Under a budget below the layers (8 GB) the tier evicts through
  prefill (3 200 evictions) and decode pays per miss: **85.8 % hit rate**
  per step (median; 17 misses of 120 routed, 42 MB, 750–800 faults of
  which most are threads finding the page present), 212–246 ms per step
  against native's 29 ms. Prefetch for the next step at depth 8 issued 48
  guesses in 62 steps (the predictor's top guesses are mostly resident
  already), 14 useful, 0 late, 0 wasted: no measurable effect (6.3 vs 6.1
  t/s), the same finding as the 480B replay.
- **Two defects found by the numbers, both in the routing report.** The
  first two tiered runs showed 100 % hits with 50 MB copied per step:
  (1) the CPU executor's per-layer routing log was a pinned-to-pinned
  `copy_`, a CPU memcpy at capture time and no graph node, so every step
  reported the capture step's ids (`c50484b`); (2) the log must be read
  after the compute stream drains (`4e7b885`); and on the server side a
  `ROUTED` burst's first line is the token boundary, which cleared the
  faulted set before the burst was scored (`134eee8`). `docs/FREETOKEN.md`
  records what a token event pairs with.
- **Where TierInfer stops and FreeToken begins** (TI-FT-003/004): the GPU
  slot cache and the pinned banks of GPU layers are FreeToken's and
  untouched; TierInfer serves only banks FreeToken would otherwise fill at
  load, for layers it decodes on the CPU; `pin()` on a tiered bank is
  refused (it would be a load), `lock()` a no-op; without `TIERINFER_SOCK`
  the patched FreeToken behaves as before.
- **Acceptance IDs moved:** TI-FT-002/005/006/008/009/010/011 → VERIFIED;
  TI-FT-003/004 VERIFIED (already); TI-FT-007 → VERIFIED (FreeToken's
  `/v1/stats` VRAM and slot cache beside TierInfer's, which holds no VRAM
  here by design); TI-FT-012 → PARTIAL (unset socket → unpatched path,
  absent tier → explicit error, refused evictions counted; failure
  injections under a running FreeToken not done). TI-PREF-008 (live) →
  VERIFIED, negative under FreeToken as in replay.

## CP-14 — the trainable prerouter on 480B routing (item 2)

- **Revision:** `1c90d9d`; `benchmarks/prerouter_eval.py` on
  `traces/qwen3coder480b-{prose,code}-400.jsonl`, run on an idle machine
  (`benchmarks/480b/raw/prerouter-eval.txt`).
- **Result:** online prerouter recall@16 77.4 % (prose) / 66.3 % (code)
  in-class and 61.7 % / 74.4 % across classes, against the adaptive blend's
  67.6 / 57.3 / 51.0 / 64.7 — ten points better in every direction; frozen
  it loses across classes. Training 4–9 s per 400-token trace on the CPU.
  V §5.1 has the table and the caveat: prefetch has shown no measurable
  effect under real compute, so this is a better guess for a mechanism
  whose value on this device is unproven.
- **Acceptance IDs moved:** TI-PRED-007 → VERIFIED.

## CP-15 — under the loader: the prerouter live, prefetch depth 8, and stripe-aligned reads (items 2, 3, 7)

- **Revision:** `00b0792` (+ `bcbdd57` merged); GLM-4.5-Air IQ4_XS, `-ngl 0
  -c 4096 -t 32`, 27 GB tier, the 111-token prompt, 32 greedy tokens, cold
  page cache, two runs per arm (`benchmarks/loader-out/glm-{exact,align512,d8-adaptive,d8-prerouter}-loader-*.json`).
  Tokens identical to native in every run.

  | arm | gen t/s (2 runs) | infer GiB | md0 reads | mean KB | prefetch issued / useful / late / wasted | copied GB | read s |
  |---|--:|--:|--:|--:|---|--:|--:|
  | exact reads, depth 0 | 0.340 / 0.382 | 68.4 | 191 k | 375 | — | 77 | — |
  | **align 512 KiB**, depth 0 | 0.390 / 0.384 | 72.0 | 172 k | 439 | — | 77 (89 read) | 90 |
  | depth 8, adaptive blend | 0.340 / 0.355 | 70.5 | 197 k | 374 | 411 / 304 / 140 / 98 | 85 | 94 |
  | depth 8, **prerouter online** | 0.193 / 0.194 | 72.1 | 202 k | 374 | 2 110 / 818 / 148 / 726 | 93 | 115 |

- **Item 7, stripe-aligned reads (`serve --align 524288`):** every read
  rounded outward to md0's 512 KiB chunk costs 15 % more bytes from the
  members (89 GiB read for 77 delivered) and gives 10 % fewer requests to
  md0 (172 k vs 191 k, 439 vs 375 KB each). Generation 0.387 vs 0.361 t/s
  median — inside the run-to-run spread of the exact arm (0.340–0.382), so
  **not shown to pay** on this device with two runs; the request count is
  the only clear effect. Left off by default; the knob and the numbers
  stay.
- **Item 3, prefetch depth 8 with the blend:** 411 guesses in 32 tokens,
  74 % useful, but generation 0.340/0.355 against 0.340/0.382 at depth 0:
  **no gain**. The guesses that were useful displaced demand reads on a
  device already at its expert-sized ceiling; the 8 GB more copied is the
  cost. Same finding as the 480B replay (V §4.2) and CP-13.
- **Item 2, the prerouter live (`--predictor prerouter`):** five times the
  guesses (its scores spread over more experts than the blend's, so more
  pass the resident filter), 818 useful (+514) and 726 wasted (+628), 4 %
  more hits, 7.5 GB more copied, 21 s more reading — and **generation
  halved** (0.193). Ten points of recall (CP-14) do not buy time when
  every extra read competes with the demand path for the same 0.75 GB/s.
  The prerouter stays available and off by default; a prefetch that pays
  needs a device with headroom, not a better guesser.
- **TI-POLICY-006:** `--adapt-depth` (halve under 35 % yield, double over
  70 %) is built and unit-tested (`tests/test_loader_policy.py`); with
  prefetch itself not paying on this device, its live A/B is not run —
  the dial would only choose between two non-gains.
- **Acceptance IDs moved:** TI-PERF-011 → VERIFIED (predictor A/B under
  the loader: negative); TI-PREF-001 → VERIFIED (asynchronous, overlapping:
  74 % of guesses land before use; measured value: none); TI-NVME-008
  extended with the align measurement; TI-POLICY-006 → VERIFIED (built,
  tested; live value bounded by the prefetch result).

## CP-16 — the server dies under FreeToken; FreeToken through FlowRunner (TI-FT-012, TI-FLOW-003)

- **Revision:** TierInfer `d030fc2`/`fe682c6`, FreeToken branch
  `tierinfer-tier` (`patches/`), FlowRunner `d556ef4`.
- **Failure injection, live** (`benchmarks/freetoken-out/flashnext-kill-tiered-2.*`,
  `freetoken_ab.py --kill-server-after 12`): Flash-Next, 12 tiered layers,
  16 GB tier; `tierinfer serve` SIGKILLed 12 s into the completion, while
  the prefill was still pulling the layers through the tier. FreeToken's
  client saw its eviction channel close, said so
  (`tierinfer-client: the server … went away; serving faults from the
  checkpoint myself`), woke every served region so the pool threads whose
  faults the dead server had already taken retried, and answered every
  later fault itself from the FTW shards, page by page. The 63 greedy
  tokens are **identical** to native and to the undisturbed 16 GB run;
  decode ran at 35.5 t/s (everything resident by then); the whole request
  took 42 s against 25 s undisturbed, the difference being page-sized
  reads for the rest of the prefill. The first attempt (`…-kill-tiered-1`)
  hung: the fallback served new faults but not the threads already asleep
  in faults the server had consumed — the wake fixed it (`d030fc2`), and
  the unit test (`tests/test_client_ftw.py`) checks the wake count.
  Without `TIERINFER_SOCK` the patched FreeToken is the unpatched one;
  with a socket that never answers, `load_ftw_banks` raises before any
  bank is allocated.
- **FreeToken through FlowRunner** (`benchmarks/flowrunner-out/ft-engine.out`):
  `flowrunner engine run` with `"runtime": "freetoken"` started
  `tierinfer serve` and `ft serve`, waited on FreeToken's health, ran one
  greedy completion (31 tokens, 24.6 s wall), read TierInfer's telemetry
  back (17.1 GB faulted in at prefill, 3 480 hits / 120 misses at decode)
  and stopped both. Two adapter defects found on the way and fixed in
  FlowRunner: a runtime that exited before health was never noticed
  (Signal(0) cannot see a zombie; every process is now reaped), and the
  runtime lacked its own bin directory and CUDA on PATH (`env` in the
  configuration; the runtime's bin dir is prepended).
- **Acceptance IDs moved:** TI-FT-012 → VERIFIED; TI-FLOW-003 → VERIFIED.

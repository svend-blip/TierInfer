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

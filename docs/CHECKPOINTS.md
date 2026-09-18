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

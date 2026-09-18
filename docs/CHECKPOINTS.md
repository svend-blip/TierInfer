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

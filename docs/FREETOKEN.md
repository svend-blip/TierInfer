# TierInfer under FreeToken

FreeToken owns the GPU tier (its slot cache) and the RAM tier (pinned or
locked host banks) and reads NVMe once, at load. What it has no notion of is
a bank that is *not* resident in RAM. TierInfer supplies exactly that, for
the layers FreeToken decodes on the CPU executor, and touches nothing else.
Design and reasons: `docs/superpowers/specs/2026-09-18-freetoken-tier-design.md`.

## What runs

```
tierinfer serve /data/ai-data/models/<ftw-dir> --sock /tmp/tierinfer-ft.sock --ram-gb 8 \
    --telemetry /tmp/ft.telemetry.jsonl

TIERINFER_SOCK=/tmp/tierinfer-ft.sock TIERINFER_SRC=<TierInfer>/src \
    ft serve --model /data/ai-data/models/<ftw-dir> --moe-strategy offload --moe-cpu-layers 12 …
```

- `tierinfer serve` on an FTW directory (one with `freetoken_weight.json`)
  reads FreeToken's index (`tierinfer.ftw.FTWIndex`): every `experts_bank`
  entry `<bank>#L<layer>` is a `[num_experts, …]` tensor whose rows are the
  experts; the dense tensors are floor. `--ram-gb` is the tier's budget for
  the served banks; there is no autoconfig for FreeToken banks yet, so it
  is required.
- FreeToken with the `tierinfer-tier` patch (a local branch of the
  `~/freetoken-qwen38` checkout; upstream is FlashML-org/FreeToken): with
  `TIERINFER_SOCK` set, the `--moe-cpu-layers` layers' banks take
  `HostResidency.TIERED`. Each such bank is a `TieredRegion`
  (`tierinfer.client`): an anonymous buffer registered with a userfaultfd
  and announced as `MAP <bank-name> <base> <len> <logical_offset>` — a slice
  of the FTW's logical byte region, not a file. FreeToken skips the
  load-time fill for those layers; the first touch of any page brings the
  whole expert (its row in every bank of that layer) from the shards;
  evictions arrive as `EVICT` and are applied with `MADV_DONTNEED`;
  `close()` sends `UNMAP` before the memory goes.
- GPU layers keep FreeToken's pinned banks and slot cache untouched.
  `pin()` on a tiered bank raises (`cudaHostRegister` would fault the whole
  bank in — a load, not a tier) and `lock()` is a no-op for the same reason.
  A flat-region bank (one entry for all layers) refuses tiering: rows must
  be addressable per layer, which the streamable converter gives.

## Routing, after the fact

llama.cpp's shim reports routing *before* a layer runs (`ROUTE`, from
`cb_eval`). FreeToken decodes inside a CUDA graph; Python is not in the step,
so the ids can only be read back afterwards. The patched CPU executor copies
each tiered layer's ids into a per-layer pinned log inside the step (captured
into the graph) and reports them before the next step as `ROUTED`. The server
then learns the routing and the token boundary as usual, and scores a hit as
"this expert was **not faulted in** during this token" — residency at report
time would say nothing, everything routed is resident by then. Prefetch on
routing works one step behind, which is what the predictor needs anyway.

Two consequences for reading the telemetry. The runtime must drain its
compute stream before reading the logs (the patch does), or the report
describes the step before the one whose faults are being scored — the
first two runs showed 100 % hits with 50 MB copied per token for exactly
that reason. And a burst's first line is the token boundary, so token
event *k* carries the faults, bytes and evictions of step *k* together with
the hit/miss scoring of step *k−1*; each number is per step, the pairing
is shifted by one. Medians over a run are unaffected.

## What to expect, and what is measured

- **Decode** faults per routed expert of the tiered layers; misses cost one
  exact read of the expert's rows (2.77 MB on Flash-Next NVFP4: 48 layers,
  512 experts, six banks per expert).
- **Prefill** of a CPU-executor layer goes through FreeToken's whole-layer
  pageable copy, which touches every row of the bank: a tiered layer is
  pulled through the tier during prefill (1.42 GB per Flash-Next layer) and
  evicted under the budget. This is recorded, not hidden; it is the same
  shape as llama.cpp's prompt batch filling the tier.
- `benchmarks/freetoken_ab.py` runs native (banks resident) against tiered
  (budget below the layers' size) on the same flags and prompt, cold cache,
  greedy, and records FreeToken's `/v1/stats` and usage, TierInfer's
  telemetry (faults, bytes, hit rate by the rule above, evictions) and the
  device counters. Results: `benchmarks/freetoken-out/` and the checkpoint
  that cites them.

## Tests

- `tests/test_ftw.py` — the FTW index on a synthetic checkpoint.
- `tests/test_client_ftw.py` — a child process through `tierinfer.client`
  against the loader: rows are the shard's bytes, one touch brings the
  whole expert across banks, a tier smaller than the layer evicts and
  re-reads correctly, `ROUTED` scores hits by faults, nothing is refused.
- The FreeToken side was checked in FreeToken's own venv against a synthetic
  checkpoint: `HostBank(backing="tierinfer")` reports `TIERED`, its tensor
  reads the shard's rows through the fault path, `pin()` is refused,
  `lock()` is a no-op, `route(after=True)` reaches the server.

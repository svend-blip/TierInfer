# An NVMe tier under FreeToken — design (item 6; DoD §4, TI-FT-*)

Written 2026-09-18 after reading FreeToken's expert-bank path
(`python/freetoken/moe/host_banks.py`, `moe/expert_banks.py`,
`moe/offload_cache.py`, `layers/moe.py`, `checkpoint/ftw.py`), before
implementing. Classified architectural.

## What FreeToken already does, and must keep doing

- Every MoE layer's experts live in a **host bank**: one `[num_experts, …]`
  tensor per layer per bank name, allocated as a lazy anonymous mmap and
  filled at load with chunked O_DIRECT reads (`HostBank`). Then settled per
  layer as `PINNED` (`cudaHostRegister`; the GPU paths copy rows from it
  by pointer inside CUDA graphs), `LOCKED` (`mlock`; CPU executor only) or
  `PAGEABLE` (CPU executor only). On this host the Human's Flash-Next
  banks are ~63 GB of pinned RAM.
- A **GPU slot cache** (`OffloadMoeCache`, `--moe-cache-size`, LRU) holds
  hot experts on the card; misses are gathered from the pinned host banks
  by `fast_index_copy` kernels. `decode_target` = gpu / cpu / hybrid
  decides per layer whether experts run on the GPU (from the slot cache)
  or on the CPU executor (from the host bank).
- Its own **FTW checkpoint format**: one logical byte region sliced into
  8 GiB shards, every tensor 4096-aligned, an index
  (`freetoken_weight.json`) naming each tensor's kind, offset and size;
  expert banks are `kind="experts_bank"` entries, one per layer per bank
  name, whose rows are the experts. Addressable to the byte, the way a
  GGUF's fused tensors are.

So FreeToken owns the GPU tier and the RAM tier, and reads NVMe once, at
load. What it has no notion of is a bank **not** resident in RAM.

## What TierInfer can add without fighting it

**The host bank becomes a TierInfer-served region for layers FreeToken
decodes on the CPU.** A `PAGEABLE` layer's bank is an anonymous buffer the
CPU executor reads; nothing pins it. If that buffer is registered with a
userfaultfd and announced to `tierinfer serve` with its identity in the
FTW region, every expert row is materialised on first touch from the FTW
shard, evicted under a budget, and prefetched on routing — exactly the
llama.cpp loader, with the FTW index in place of the GGUF index. FreeToken
skips the load-time fill for those layers (the tier fills them), and its
own GPU slot cache and pinned layers are untouched. Where FreeToken's
mechanisms already do the job (PINNED layers, the slot cache), TierInfer
does nothing; where FreeToken has no mechanism (a bank that does not fit
in RAM), TierInfer supplies one. That is DoD §4's "integrate with those
mechanisms rather than duplicate or fight them", made concrete.

**Why not PINNED layers.** `cudaHostRegister` pins pages, which forces them
present: registering a userfaultfd region faults the whole bank in at
registration — a full load, not a tier. So the GPU-offload layers stay
FreeToken's; TierInfer's tier applies to the CPU-executor layers
(`--moe-cpu-layers` selects them; `decode_target` cpu/hybrid). This is a
property of CUDA, recorded rather than worked around.

## Pieces

1. **`tierinfer.ftw`** — the FTW layout inspector: reads
   `freetoken_weight.json`, resolves logical offsets to (shard file,
   offset), and exposes expert rows per (layer, expert) for every bank
   name plus the dense tensors as floor. Same shape of answer as
   `tierinfer.index.ModelIndex` (regions with keys), so the loader's
   `FileLayout` takes either.
2. **Protocol** — `MAP <tag> <base> <len> [<logical_offset>]`: a mapping
   that is not a file but a slice of a logical region. The server maps
   buffer offset → logical offset → shard file + offset through the FTW
   index. The shim is unchanged; FreeToken's side speaks the protocol from
   Python (a small client in `host_banks.py`: register uffd, connect,
   announce, receive evictions).
3. **FreeToken patch, minimal** — `HostBank(backing="tierinfer")` and a
   `HostResidency.TIERED` label: allocate lazy anon mmap + `MADV_NOHUGEPAGE`,
   register with userfaultfd (`UFFD_USER_MODE_ONLY`), announce to
   `TIERINFER_SOCK` with the entry's logical offset, **skip the fill**, and
   an eviction thread applying `MADV_DONTNEED`. `load_ftw_banks` honours
   the label for CPU-executor layers only and refuses it for GPU layers
   with the reason above. Routing: the CPU executor's decode path knows
   `expert_ids` per layer per step; a one-line hook sends `ROUTE` to the
   server so the tracker, predictor and prefetch see real routing.
4. **Telemetry** — TierInfer's loader telemetry for the tiered layers
   (hits, faults, bytes, evictions, prefetch) beside FreeToken's
   `/v1/stats` (its own slot-cache hits, decode t/s) through the existing
   adapter's `normalise`, in one run record: FreeToken-native behaviour and
   TierInfer decisions are separate namespaces (TI-FT-009).
5. **A/B (TI-FT-011)** — Flash-Next NVFP4 (121 GB FTW, the Human's model)
   with N layers on the CPU executor: native (those layers' banks fully
   resident and pageable) against tiered (the same layers served by
   TierInfer under a budget below their size), same prompt, greedy tokens
   compared, decode t/s from FreeToken, bytes/faults from TierInfer,
   device counters from `/proc/diskstats`. Repeated.

## Order and gates

Build 1 and 2 with tests against a synthetic FTW checkpoint (the format is
documented and small to write). Then 3 on a branch of the FreeToken
checkout, behind the new residency label so an unpatched configuration
behaves exactly as before. Then 5. If the CPU-executor path's row reads do
not fault the way llama.cpp's do (e.g. FreeToken touches whole banks in a
way that defeats per-expert service), that is measured and recorded, and
the design stops there with the reason.

# The llama.cpp loader — design (goal 6)

Written 2026-09-18 before implementation, under the Human's mandate to build
items 1–7 of the post-addendum list autonomously. Classified architectural:
a new subsystem that changes how TierInfer meets the runtime.

## The question it answers

Every TierInfer mechanism was measured under *replayed* routing because
nothing put its buffers under a running model (`docs/AUDIT-2026-09-18.md`,
item 12). Goal 6 as the scope words it: real inference while TierInfer
controls weight residency and streaming, benchmarkable against native
mmap, with the clean baseline preserved.

## What was measured before choosing

1. **Advice does nothing** (`REAL-ROUTING.md`): `posix_fadvise` cannot move
   pages a live file mapping holds. So the loader has to *own* the memory
   the model is read from, not advise around it.
2. **The routing is observable without a patch** (`tools/trace`):
   `cb_eval` on `ffn_moe_topk-<layer>` through the public API.
3. **A process can serve its own page faults here, unprivileged**
   (`uffd_probe.py`, 2026-09-18): `userfaultfd(O_CLOEXEC|O_NONBLOCK|
   UFFD_USER_MODE_ONLY)` succeeds with `vm.unprivileged_userfaultfd=0`; an
   anonymous `MAP_PRIVATE` region registered for MISSING faults; on the
   first touch of any page of a 32 MB "expert" the handler `UFFDIO_COPY`s
   the **whole 32 MB** in ~10 ms and the toucher continues; `MADV_DONTNEED`
   on the slab drops it (RSS falls) and the next touch faults again. That is
   exactly the tier the scope asks for: expert-granular residency, explicit
   eviction, no kernel heuristics in the loop.

## Approaches considered

**A. Patch llama.cpp's model loader** to fetch expert slabs from TierInfer
before `ggml_mul_mat_id`. Most direct; a fork of the loader and of the MoE
graph path, re-done per llama.cpp version; the clean baseline becomes a
build flag. Rejected for now: the scope's "adapters may patch" allows it,
but A and B reach the same residency control and B leaves llama.cpp's
binaries untouched.

**B. Own the mapping from outside** (chosen). An `LD_PRELOAD` shim
interposes `mmap` in the llama.cpp process: for the model's files it
returns an anonymous region registered with a userfaultfd instead of a
file mapping, and hands the descriptor to a TierInfer server over a unix
socket. Every touch of a non-resident expert becomes one fault the server
answers with the whole expert; eviction is an `madvise(MADV_DONTNEED)`
issued by the shim on the server's instruction; routing reaches the server
through a `cb_eval` the shim installs by interposing
`llama_init_from_model`. Native = run without the shim. Nothing in
llama.cpp changes.

**C. Serve faults from inside llama.cpp** (shim only, no server): simpler
plumbing, but the policy, cache, predictor and telemetry would have to be
rewritten in C inside a preload library. Rejected: the scope keeps the core
runtime-independent and the existing Python components are the ones that
were measured.

## Design

```
llama-server / llama-cli / tierinfer-trace  (unmodified binaries)
   │  LD_PRELOAD=libtierinfer_mmap.so   TIERINFER_SOCK=/run/user/…/tierinfer.sock
   │
   ├─ mmap(model shard)  ──►  anonymous region + userfaultfd(MISSING)  ──┐
   ├─ llama_init_from_model ─► cb_eval installed: topk rows → socket     │ SCM_RIGHTS: uffd,
   └─ evict thread: reads "evict off len" from socket → madvise(DONTNEED) │ path, base, size
                                                                          ▼
tierinfer serve  (Python, one process, N fault workers)
   ├─ index: (shard, offset) → expert (layer, e) | floor range
   ├─ RAM tier: which experts are materialised, bounded by autoconfig's budget;
   │            ExpertCache bookkeeping (payload None, size = slab), LRU-equal
   ├─ fault: expert slab → StorageBackend/ExpertStreamer read → UFFDIO_COPY whole slab
   │         floor range → 8 MB aligned chunk, pinned
   ├─ routing events → tracker + predictor → prefetch = UFFDIO_COPY before the touch
   ├─ over budget → victim → "evict" to the shim → slab gone → next touch faults
   └─ telemetry: residency hits (routed & resident), faults, bytes, prefetch
                 useful/late/wasted, evictions; md0 counters; llama.cpp's t/s
```

**What the RAM tier is:** the set of experts currently materialised in the
llama.cpp process's mapping. The server holds no second copy — only pooled
read buffers in flight — so RAM is paid once. Its size is `autoconfig`'s
RAM budget; over it, the policy evicts.

**What the VRAM tier is under this design:** llama.cpp's, via `-ngl` /
`-ncmoe`. Its kernels cannot consume `VramResidency`'s allocations, and the
scope's goal 9 says "where runtime integration permits explicit control";
for llama.cpp it does not, and this design says so rather than pretending.

**Routing:** the shim's `cb_eval` sends `(layer, n_tokens, experts…)` per
routing tensor, read row by row (the strided-view lesson). Hits are
measurable exactly: a routed expert that is resident when its routing
arrives is a hit; one that is not will fault. Prefetch has real lead time:
attention of layer L runs between L's routing and L's FFN, and the
predictor speaks for L+1 before that.

**Correctness:** the bytes copied are the file's bytes for that offset
(`StorageBackend.read` on the index's ranges; `exact_load` semantics), and
the acceptance test is stronger than any checksum: **greedy decoding under
the loader must produce the same tokens as native** on the same prompt.

**Failure:** a fault the server cannot serve (I/O error) is answered by
`UFFDIO_COPY` of the bytes read through the exact path or, if that fails,
the server logs and exits — the faulting thread then blocks forever, which
is the honest outcome of a model whose bytes cannot be read, and llama.cpp
never sees wrong weights. A dead server leaves llama.cpp stuck, not wrong;
the shim's socket loss is logged.

## Scope of the first build

- GLM-4.5-Air IQ4_XS (single file, 47 layers) under a 32 GB ceiling — the
  scope's original target, native measured at 0.80 t/s there.
- Qwen3-Coder-480B (six shards) at `-ngl 99 -ncmoe 60` — the addendum's
  model, native 0.42–0.46 t/s.
- Same prompt, 3 runs each, native vs loader; tokens identical; md0 bytes
  and reads per token; residency hit rate; prefetch useful/late/wasted.

Out of scope for the first build: a VRAM expert tier under llama.cpp
(impossible without a patch), FreeToken (item 6, separate design),
priority classes and stripe alignment (items 3 and 7, measured under this
loader once it runs).

## Files

- `tools/uffd/tierinfer_mmap.c`, `tools/uffd/build.sh` — the shim.
- `src/tierinfer/loader.py` — the server: `MappingRegistry`, `FaultServer`,
  `ResidencyTier`, routing intake, telemetry; `tierinfer serve` in `cli.py`.
- `benchmarks/loader.py` → `benchmarks/LOADER.md` — the A/B harness and
  result.
- Tests: the shim against a small synthetic "model" process; the server's
  fault path with a synthetic GGUF and a child process that touches it.

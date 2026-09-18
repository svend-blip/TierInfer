# TierInfer

VRAM, system RAM and NVMe as **one adaptive inference memory hierarchy** for
large local models.

Storage is normally where a model is read from once. TierInfer treats NVMe as
a third memory tier that stays active during generation, so a model larger
than VRAM and RAM can be served without falling back to whatever the operating
system's demand paging happens to do.

The target is sparse Mixture-of-Experts inference, where only a few experts of
each layer run for any given token — so most of the model does not need to be
anywhere expensive.

## The number the project turns on

Measured on the reference model, GLM-4.5-Air-Derestricted IQ4_XS
(`glm4moe`, 47 layers, 128 experts per layer, 8 used per token):

| | |
|---|---:|
| Model on disk | **56.46 GB** |
| Always resident — attention, norms, router, shared experts | **4.55 GB** |
| Routed experts | 51.91 GB |
| Working set for one token, from the layout | 8.13 GB |
| **Working set for one token, measured from real routing** | **7.73 GB** |

7.73 GB of 56.46 GB is what a token actually needs — **13.7 %** — measured by
capturing 800 tokens of routing out of the running model, not inferred from
the file. That is a quarter of a 32 GB card. The whole project is the distance
between those two numbers.

And the distance is real: llama.cpp given a 32 GB ceiling on this model runs
at **0.80 tokens/s instead of 5.90**, and reads **113.6 GB for a 56.5 GB
file**, because it keeps whatever it has touched rather than what it is about
to need.

## Status

**Real, measured components, and now the spine.** Every mechanism between
the GGUF directory and the GPU exists, is tested against real files and a
real card, and has numbers behind it; `python tools/smoketest.py` exercises
each against the real model, the real NVMe and the real GPU in about ten
seconds. Since 2026-09-18 they also sit under a running model: the loader
(`docs/LOADER.md`) is a preloaded shim that turns llama.cpp's file mappings
into userfaultfd regions the TierInfer server answers per expert, with no
change to llama.cpp. On Qwen3-Coder-480B-A35B Q4_K_M from a USB RAID0 it
generates **0.647 t/s against native's 0.467** (three cold runs each, greedy
tokens identical in all six, 36 % fewer bytes in 5× fewer reads; CP-12 in
`docs/CHECKPOINTS.md`). The same server serves FreeToken's expert banks to
its CPU-executor layers through a Python client (`docs/FREETOKEN.md`).
`docs/AUDIT-2026-09-18.md` is the component-by-component account,
`docs/ACCEPTANCE-STATUS.md` the live status per acceptance criterion
(2026-09-18: 119 of 120 verified or accepted, one blocked by design).
Everything below is a measurement on the reference model, not a plan.

| | |
|---|---|
| **GGUF layout and expert addressing** | `load expert 37 of layer 18` returns byte ranges, not a hope about page faults |
| **Baseline benchmark** | `benchmarks/BASELINE.md` — warm, cold, and two ceilings, with NVMe counters and page-cache residency, none of it needing root |
| **NVMe→RAM streaming** | `benchmarks/STREAMING.md` — 3.47× demand paging at eight workers |
| **Bounded expert cache** | value policy, a revalidating heap, and an exact LRU front |
| **Routing capture from llama.cpp** | `tools/trace` — no patch, through the public `cb_eval` hook |
| **Expert prediction** | four predictors and a recall@k harness, scored on real routing |
| **Async prefetch** | the one place a guess causes I/O, and a miss always falls back to an exact read |

### What the measurements changed

Two results are worth the front page because they overturned what this
project assumed about itself.

**The cache module was wrong, and real routing said so.** It opened by
asserting that least-recently-used was the wrong default for MoE. Against a
synthetic Zipf trace that held by up to 13 points of hit rate; against 800
tokens captured out of GLM-4.5-Air it did not. Configured as it now is, the
policy *equals* LRU rather than beating it. Three defects had to be fixed
before it even got that far, and every one was invisible against synthetic
data. `benchmarks/REAL-ROUTING.md` has them.

**Prediction was better than the synthetic trace suggested, by a factor of
ten.** Context beats the frequency floor by 12–14 points on real routing, not
one. The same synthetic generator understated one thing and overstated the
other, for the same reason: it was written by the same hand as the code it
was measuring.

### The horizon, which decides the architecture

How much of the file does a window of W consecutive tokens need?

| window | of the file |
|-------:|------------:|
| 1 token | 13.7 % |
| 2 | 18.7 % |
| 8 | 37.6 % |
| 32 | **64.9 %** |
| 400 | 93.0 % |

A 32-token window needs 36.65 GB, and the ceiling that collapsed throughput
7.4× was 32 GB. That is the mechanism, not a coincidence.

It also settles where the work belongs. The needed set doubles by the second
token, so anything managing residency from *outside* the inference loop
cannot act on the horizon the data actually has. Tested rather than assumed:
a helper process holding the 4.55 GB floor warm bought 12.5 % for 34 GB of
extra reads (`benchmarks/residency.py`). The decision has to be made between
layers.

### One thing the layout decides for us

In this model the routed experts of a layer are **fused into one tensor per
projection** — `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` — with the
expert index as the last dimension. An expert is therefore a slice inside a
tensor, not a tensor of its own, and its byte range is arithmetic on the fused
tensor's offset. Where that arithmetic does not divide evenly, TierInfer
refuses rather than guesses: a wrong range reads the wrong weights, which is
worse than no range at all.

Not every MoE build stores experts this way. The inspector reports what it
finds rather than assuming a layout.

### Reading an expert on purpose

`tierinfer.storage` reads named byte ranges and times every one. Measured on
the reference host (Samsung 990 PRO), the same 8.94 MB expert, median of five:

| Mode | Time | Bandwidth | Operations |
|---|---:|---:|---:|
| Cold, one read per projection | 13.45 ms | 0.70 GB/s | 3 |
| Cold, coalesced | 14.56 ms | 0.64 GB/s | 3 |
| **Cold, 4 KB pages** | **54.23 ms** | 0.17 GB/s | **2 288** |
| Warm, one read per projection | 1.95 ms | 4.81 GB/s | 3 |

Asking for the same bytes 4 KB at a time costs **4× the time and 763× the
operations**. That is the baseline experiment's pathology reproduced in
isolation, and the reason the rest of the system exists.

Reproduce it with `python benchmarks/read_paths.py <model.gguf>`. Cold modes
evict their own ranges from the page cache first — `posix_fadvise` needs no
privileges, so no cache has to be dropped system-wide to get an honest number.

## What is not measured yet

Prefetch depth above zero under the loader (the 480B A/B ran at depth 0;
the classes and coalescing are built and unit-tested), the trainable
prerouter's recall on the 480B traces, stripe-aligned reads on md, a
TierInfer VRAM tier under llama.cpp (llama.cpp's own `-ncmoe` owns the card;
a TierInfer VRAM tier there needs a llama.cpp patch, documented and not
attempted), and FlowRunner's end-to-end run with real binaries (the engine
adapter exists on the FlowRunner side). The advisory route was tried first
and measured to do nothing (`benchmarks/REAL-ROUTING.md`): a live mapping
keeps its pages whatever `posix_fadvise` is told.

`SCOPE.md`, `docs/DEFINITION-OF-DONE.md` and `docs/ACCEPTANCE-MATRIX.md`
carry the plan, the bar, and what each criterion has measured so far.

## The 480B validation (2026-09-18)

`docs/SCOPE-ADDENDUM-480B.md` put the components under Qwen3-Coder-480B-A35B
Q4_K_M — six shards, 270 GiB, on a USB RAID0 — with `docs/AUDIT-2026-09-18.md`
first and `benchmarks/480b/VALIDATION-480B.md` as the result. In one line
each: native llama.cpp generates at 0.24 t/s CPU-bound on page faults with
the device half idle; TierInfer's RAM tier under the same routing reads
**36–65 % fewer bytes in 20–40× fewer operations with less RAM**; prefetch
and prediction add nothing measurable; the VRAM tier catches a fifth of the
traffic; 18 000 delivered experts under injected failures matched the file
byte for byte; and, once the loader existed, llama.cpp with TierInfer
generated 39 % faster than native on the same flags with identical tokens
(§4.4).

## Why this is worth doing

Measured beforehand on this workstation (RTX 5090 32 GB, 187 GB RAM, Samsung
990 PRO), with llama.cpp at `-ngl 25`:

| Condition | Decode |
|---|---:|
| Unconstrained RAM | **7.7 tok/s** |
| Forced NVMe paging (`MemoryMax` 52 GB) | **5.5 tok/s** |

Generation survived the memory limit — at the cost of roughly 200 000 reads
per second of about 4 KB each. Storage *can* back inference; generic paging
just does it badly, reacting after a page is already missing.

TierInfer's job is to replace that with reads it asked for in advance, at
expert granularity, while the GPU is busy with the previous token.

## Using it

Three ways in, all on the same server process. Details, flags and
diagnostics: `docs/LOADER.md` (llama.cpp), `docs/FREETOKEN.md` (FreeToken),
FlowRunner's `docs/ENGINE-tierinfer.md` (through an orchestrator).

**1. Under llama.cpp, unmodified.** Build the shim once against your
llama.cpp tree, start the server on any shard of the model, preload the
shim into `llama-server`:

```console
LLAMA_CPP=~/llama.cpp tools/uffd/build.sh            # once: build/libtierinfer_mmap.so

tierinfer serve model-00001-of-00006.gguf --sock /tmp/tierinfer.sock --ram-gb 150 \
    --telemetry run.jsonl                             # terminal 1; prints the env for terminal 2

LD_PRELOAD=$PWD/build/libtierinfer_mmap.so TIERINFER_SOCK=/tmp/tierinfer.sock \
TIERINFER_FILES=model-00001-of-00006.gguf:…:model-00006-of-00006.gguf \
llama-server -m model-00001-of-00006.gguf -ngl 99 -ncmoe 60 -c 4096   # terminal 2
```

The shim turns each shard's mapping into a region the server fills per
expert on first touch and empties under `--ram-gb`; routing reaches the
server through `cb_eval`. `--ram-gb` omitted means autoconfig's share of
available RAM. `--depth N` turns prefetch on, `--predictor prerouter`
swaps the guesser, `--align 524288` rounds reads to md's stripe,
`--adapt-depth` lets the depth follow prefetch yield — every one of
those is measured and none pays on this host (CP-13, CP-15), so the
defaults are demand-only and exact. `TIERINFER_DEBUG=1` on the server
logs every fault. Without `LD_PRELOAD` the same binary is the native
baseline.

**2. Under FreeToken.** Apply `patches/freetoken-tierinfer-tier.patch` to
a FreeToken checkout, serve the FTW checkpoint directory, and tell
FreeToken where the socket is; the `--moe-cpu-layers` layers' banks
become TierInfer-served buffers, GPU layers stay FreeToken's:

```console
tierinfer serve /models/<ftw-dir> --sock /tmp/tierinfer-ft.sock --ram-gb 16 --telemetry ft.jsonl
TIERINFER_SOCK=/tmp/tierinfer-ft.sock TIERINFER_SRC=$PWD/src \
    ft serve --model /models/<ftw-dir> --moe-strategy offload --moe-cpu-layers 12 …
```

If the server dies, FreeToken's client serves its own faults from the
checkpoint and says so; the output stays exact.

**3. Through FlowRunner.** A flow declares a capability document (model,
context, residency share) or, for FreeToken, a tier size and the tiered
layers; `flowrunner engine run --config engine.json --complete "…" --once`
resolves it with `tierinfer resolve`, starts server and runtime, runs the
completion and hands back the endpoint, the loaded model, timings and
TierInfer's telemetry. Both runtimes are supported (`runtime: freetoken`).

**Looking at a model without running anything:**

```console
tierinfer inspect model-00001-of-00006.gguf        # layers, experts, sizes, shards
tierinfer expert  model-00001-of-00006.gguf 18 37  # the byte ranges of one expert
tierinfer resolve capability.json model-00001-of-00006.gguf   # what this host would grant
```

**Measuring:** `benchmarks/loader_ab.py` (llama.cpp native vs loader, cold,
repeated, greedy tokens compared), `benchmarks/freetoken_ab.py` (FreeToken
native vs tiered, with `--kill-server-after` for the failure injection),
`benchmarks/replay.py` (real routing, real files, no compute),
`benchmarks/prerouter_eval.py`, `tools/trace` (routing capture), and
`python tools/smoketest.py` for the ten-second check that every component
works on this machine. Results and the method live in
`benchmarks/480b/VALIDATION-480B.md`, `benchmarks/loader-out/`,
`benchmarks/freetoken-out/` and `docs/CHECKPOINTS.md`.

## Install

```console
pip install -e .
```

Python 3.10+, no dependencies. Run the tests with `pytest`; none of them need
a model file.

## Licence and origin

The originating inspiration is [Edge0](https://github.com/Edge0-AI/Edge0),
which demonstrated streaming MoE inference with experts left on SSD.
TierInfer generalises that idea into a VRAM/RAM/NVMe hierarchy with its own
abstractions, measurements and runtime adapters.

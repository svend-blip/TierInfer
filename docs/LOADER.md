# Running llama.cpp with the TierInfer loader

What exists today, how to run it, how to see that it is running, and how to
turn it off. Nothing here describes planned behaviour.

## What it is

An unmodified `llama-server` / `llama-cli` (any binary linked against
`libllama.so`) is started with `libtierinfer_mmap.so` preloaded. The shim
gives it an anonymous, userfaultfd-registered region in place of each model
file's mapping and connects to `tierinfer serve`. The first touch of any
page of an expert is one fault the server answers with the whole expert,
read from the file at the index's offsets; the set of materialised experts
is TierInfer's RAM tier, bounded by a byte budget and evicted by its policy;
routing reaches the server through a `cb_eval` the shim installs. Design
and measurements: `docs/superpowers/specs/2026-09-18-llama-loader-design.md`,
`docs/CHECKPOINTS.md` CP-11 onward.

The VRAM tier under llama.cpp is llama.cpp's own (`-ngl`, `-ncmoe`).
TierInfer's `VramResidency` cannot be consumed by llama.cpp's kernels
without a patch to llama.cpp, and this integration makes none.

## Build

```console
pip install -e .                       # TierInfer, no dependencies
LLAMA_CPP=~/llama.cpp-qwen38 tools/uffd/build.sh
```

`build.sh` compiles the shim against the headers and shared objects of an
existing llama.cpp build; nothing in that tree is changed. Requirements:
Linux with `userfaultfd` (kernel ≥ 5.11; `UFFD_USER_MODE_ONLY` lets an
unprivileged process use it even with `vm.unprivileged_userfaultfd=0`), a
GGUF model (split models are read as one), and a llama.cpp built as shared
libraries.

## Run

Terminal 1 — the server:

```console
tierinfer serve /path/to/model-00001-of-00006.gguf --sock /tmp/tierinfer.sock \
    [--ram-gb 150] [--workers 8] [--depth 0] [--telemetry run.jsonl]
```

`--ram-gb` is the RAM tier; without it `autoconfig` derives one (60 % of
available RAM minus the model's floor). `--depth` is the prefetch depth per
layer (0 = demand only; every measured value so far says leave it 0 — see
CP-12). The server prints the exact environment to give the runtime.

Terminal 2 — the runtime, unmodified:

```console
LD_PRELOAD=/path/to/TierInfer/build/libtierinfer_mmap.so \
TIERINFER_SOCK=/tmp/tierinfer.sock \
TIERINFER_FILES=/path/to/model-00001-of-00006.gguf:/path/to/model-00002-of-00006.gguf:... \
llama-server -m /path/to/model-00001-of-00006.gguf -ngl 99 -ncmoe 60 -c 4096
```

`TIERINFER_FILES` lists every file the shim should take over (the server
prints the list). Files not listed are mapped natively.

## How to know it is active

- The runtime's stderr says `tierinfer-mmap: serving <file> through
  userfaultfd (...)` for the first file, and `routing callback installed`.
  If it says `standing aside (native mmap)`, it is native — the socket was
  unset or unreachable — and every number you measure is llama.cpp's.
- The server logs `pid N: serving <file> (...), <k> regions` per file, then
  one line per token (hit rate, faults, bytes copied, evictions, prefetch
  issued/useful/late/wasted, resident set).
- The runtime's RSS stays at the tier budget plus the floor; native's grows
  with the file.
- `kill -USR2 <server pid>` prints the counters; `kill -USR1` dumps every
  thread's stack.

## Telemetry

`--telemetry run.jsonl` writes the shared schema (`tierinfer.telemetry`):
`run.open`, one `mapping` event per file, one `token` event per generated
token (hits, misses, faults, bytes copied, evictions, prefetch counters,
wall ms, resident set), a snapshot every ten tokens, `run.close`. Read it
with `tierinfer.telemetry.read` / `deltas`. `benchmarks/loader_ab.py`
measures native against the loader on the same prompt and checks that the
greedy tokens are identical.

## Diagnosing

- A runtime thread stuck in `handle_userfault` (`cat /proc/<pid>/task/*/wchan`)
  with an idle server is a fault the server did not answer. The server
  shouts `FAULT AT ... NOT SERVED` with the exception and stops the worker
  rather than answer with anything else; the runtime is left waiting, never
  given wrong bytes.
- A page faulting more than 64 times ends the worker with a message naming
  the offset and region.
- `TIERINFER_DEBUG=1` on the server logs every fault and copy (large).

## Turning it off

Start the runtime without `LD_PRELOAD` (or without `TIERINFER_SOCK`). That
is the native baseline, byte for byte the same binary.

## Known limits

- One server serves one model; several runtimes may share it.
- The prompt's working set can exceed the tier (a 111-token prompt on
  GLM-4.5-Air touches ~50 GB); the tier thrashes through the prompt as a
  page cache would under the same ceiling.
- Each fault stalls the faulting thread for one expert read (~10–20 ms);
  32 compute threads fault in parallel and 8 workers serve them. The
  measured cost against native is in CP-12.

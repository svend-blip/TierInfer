# Architecture

```text
                        FlowRunner
                            │
                            ▼
                     TierInfer Core
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
  Model Inspector       Tier Manager         Telemetry
       │                    │
       │          ┌─────────┼─────────┐
       │        VRAM       RAM       NVMe
       │          └────┬────┴────┬────┘
       │          Expert Cache   │
       │          Expert Tracker │
       │          Expert Predictor
       │          Async Prefetcher
       └───────────────┬┘
             ┌─────────┴─────────┐
         llama.cpp            FreeToken
             └─────────┬─────────┘
                       ▼
                   Local Model
```

## Model Inspector — implemented

`tierinfer.gguf` reads a GGUF file's header, metadata and tensor directory
without reading weights. `tierinfer.index` classifies each tensor and answers
for byte ranges.

### What the GGUF directory gives us

For every tensor: name, shape, ggml type, and an offset relative to the data
section. The data section begins at the first aligned byte after the
directory. So `file_offset = data_offset + offset`, and a tensor's size comes
from its type's block layout — quantized types store whole blocks of 32 or
256 elements.

That size arithmetic is worth checking rather than trusting, and there is a
cheap check: the last tensor must end exactly at the end of the file. On the
reference model it does, at byte 60 630 797 344.

### Classification

Per layer, tensors fall into groups that TierInfer must treat differently:

| Group | Example | Residency |
|---|---|---|
| attention | `blk.N.attn_q.weight` | always |
| norms | `blk.N.attn_norm.weight` | always |
| router | `blk.N.ffn_gate_inp.weight` | always — it decides what to load |
| shared expert | `blk.N.ffn_gate_shexp.weight` | always — every token uses it |
| routed experts | `blk.N.ffn_gate_exps.weight` | on demand |

The first four are the floor: 4.55 GB on the reference model. The fifth is
51.91 GB and is where the tiering happens.

### Expert addressing

The routed experts of a layer are fused into one tensor per projection, with
the expert index as the last dimension. Expert *i* is therefore the *i*-th
contiguous slab, and

```text
range = fused.file_offset + i * (fused.nbytes / expert_count)
```

when — and only when — `fused.nbytes` divides evenly by the expert count and
the last dimension equals it. Both are checked; failure raises rather than
returns an approximate range.

One expert of the reference model is 8.94 MB across its three projections.
That is the unit of transfer the rest of the system is built around: large
enough that a read is efficient, small enough that 8 per layer is 71 MB
rather than a gigabyte.

## Not yet implemented

The Tier Manager, caches, tracker, predictor, prefetcher, storage backend,
telemetry and adapters. `SCOPE.md` carries the order.

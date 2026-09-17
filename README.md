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
| **Working set for one token** | **8.13 GB** |

8.13 GB of 56.46 GB is what a token actually needs. That is a quarter of a
32 GB card. The whole project is the distance between those two numbers.

## Status

Early. What works today:

- **GGUF layout inspection.** The directory of a 56.5 GB file is read in
  0.20 s without touching a weight, and every tensor's computed size checks
  out: the last tensor ends at byte 60 630 797 344, which is exactly the file
  size.
- **Expert addressing.** `load expert 37 of layer 18` returns byte ranges
  rather than a hope about page faults.

```console
$ tierinfer inspect models/GLM-4.5-Air-Derestricted.IQ4_XS.gguf
architecture     glm4moe (GGUF v3, 803 tensors)
layers           47, of which 46 are MoE
experts          128 per layer, 8 used per token, 8.94 MB each

total               56.46 GB
always resident      4.55 GB   attention, norms, router, shared experts
routed experts      51.91 GB
working set          8.13 GB   what one token actually needs

$ tierinfer expert models/GLM-4.5-Air-Derestricted.IQ4_XS.gguf 18 37
layer 18 expert 37: 8.94 MB in 3 ranges
  blk.18.ffn_gate_exps.weight#expert37   offset    23722005536  +  3063808
  blk.18.ffn_up_exps.weight#expert37     offset    24119333920  +  3063808
  blk.18.ffn_down_exps.weight#expert37   offset    23310193696  +  3244032
```

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

## What is not built yet

Everything after indexing: the tier manager, the RAM and VRAM caches, the
expert activity tracker, the predictor and prerouter, the async prefetch
engine, the storage backend, the telemetry, and the llama.cpp, FreeToken and
FlowRunner adapters. `SCOPE.md` carries the full plan and its order.

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

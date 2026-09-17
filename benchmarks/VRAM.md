# VRAM — running a 56 GB model on a 32 GB card

Measured 2026-09-17 on an RTX 5090 (31.36 GB usable), with the routing traces
from `tools/trace`. Transfers are real: pinned host memory, `cudaMemcpy`,
timed. The bytes are not the model's own — replaying a trace needs the right
sizes and the right sequence, not the right weights — but the cost of moving
them is.

```
python benchmarks/vram.py MODEL.gguf TRACE.jsonl --context 16384
```

## The budget, derived rather than guessed

    weights = total − reserve − kv_cache − runtime_overhead

The KV term comes from the model's own metadata, and it is **verified against
llama.cpp's own allocation**:

| context | this formula | llama.cpp reports |
|--------:|-------------:|------------------:|
| 2 048 | 376 MiB | 376 MiB |
| 8 192 | 1 504 MiB | 1 504 MiB |
| 16 384 | 3 008 MiB | 3 008 MiB |

Identical at every point. Grouped-query attention is why it is small enough
to be worth computing: this model has 96 query heads against 8 key/value
heads, so using the head count would overstate the cache twelvefold.

`runtime_overhead` is *not* derived — it belongs to the runtime, not the
model — so it is measured instead. llama.cpp reports 330, 328 and 320 MiB at
those same three contexts, near enough constant, which also shows it scales
with the batch rather than the context. The default is 512 MB, above every
measurement, and labelled as one runtime's number rather than a law.

## What fits, and what the misses cost

Trace B (a code-generation prompt, 400 generated tokens, 360 expert
activations each) against a pool sized by the budget:

| context | KV | for weights | slots | resident | hit rate | transfers/token | ms/token | added to a warm token |
|--------:|---:|------------:|------:|---------:|---------:|----------------:|---------:|----------------------:|
| 4 096 | 0.73 GB | 29.12 GB | 2 523 | 24.6 GB | 79.3 % | 75 | 28.6 | **17 %** |
| 8 192 | 1.47 GB | 28.38 GB | 2 447 | 23.8 GB | 78.8 % | 76 | 29.3 | 17 % |
| 16 384 | 2.94 GB | 26.91 GB | 2 296 | 22.4 GB | 76.4 % | 85 | 32.5 | 19 % |
| 32 768 | 5.88 GB | 23.98 GB | 1 995 | 19.4 GB | 71.5 % | 103 | 39.5 | 23 % |
| 65 536 | 11.75 GB | 18.10 GB | 1 391 | 13.5 GB | 61.3 % | 139 | 53.7 | 32 % |
| 131 072 | 23.50 GB | 6.35 GB | 184 | 1.8 GB | **0.0 %** | 360 | 140.8 | 83 % |

A warm unconstrained token takes 169 ms (5.90 t/s, `BASELINE.md`), so the
last column is what the transfers would add to it.

**At ordinary contexts the answer is about a fifth.** 22–25 GB of experts
stay resident, three quarters of what a token asks for is already on the
card, and the 75–85 that are not cost 29–33 ms. For comparison, the same
model under a 32 GB *RAM* ceiling ran 7.4× slower, not 1.2×.

**A cache below one token's working set does not degrade — it collapses.**
At 131 072 tokens of context the KV cache takes 23.50 GB and leaves room for
184 experts. A token needs 360. So every expert is evicted before the next
token asks for it and the hit rate is not low, it is zero. Long context and
expert residency compete for the same card, directly, and the crossing point
is sharp.

## What it cost to move a byte

| transfer | rate | per 9.97 MB expert |
|----------|-----:|-------------------:|
| pageable host → device | 16.0 GB/s | 0.62 ms |
| **pinned host → device** | **27.6 GB/s** | **0.36 ms** |

Pinning is worth 1.7×, and the pool takes pinned memory for that reason. For
scale against the tier below it: NVMe → RAM streams at 3.0 GB/s with eight
workers (`STREAMING.md`), which is 3.3 ms for the same expert — **nine times
slower**. The tiers are a decade apart in cost and the policy should treat
them that way.

## What this does not measure

**No compute overlaps the transfers.** Every figure here is the transfer cost
in isolation. A real implementation would issue them on a copy stream while
the previous layer computes, so the 29 ms at 16 k context is a ceiling on the
cost, not a prediction of it.

**The pool is allocated once and never resized.** That is deliberate — putting
`cudaMalloc` in the token loop would fragment the heap and add a syscall to
every miss — but it means the budget has to be right before the run starts,
not adjusted during it. Goal 10 is where that becomes adaptive.

**One card, one model, one quantisation.** The shape of the curve should
generalise; the numbers on it are this card's.

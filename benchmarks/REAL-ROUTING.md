# Real routing — what the model actually does, and what it overturns

Until this, every claim in this project about *which experts a token needs*
was measured against a trace this project also generated. That proves the
code runs. It proves nothing about a model.

`tools/trace` captures the real thing. Two traces of 400 generated tokens
each were taken from GLM-4.5-Air-Derestricted IQ4_XS on 2026-09-17, with
different prompts so that a conclusion drawn on one could be checked on the
other:

| trace | prompt | tokens | layers | routed | distinct experts seen |
|-------|--------|-------:|-------:|-------:|----------------------:|
| A | prose: explain MoE routing | 440 | 45 | 8 | 128 of 128 |
| B | code: rewrite a function, then test it | 441 | 45 | 8 | 128 of 128 |

All 128 experts of every layer appear within the first nine tokens. There is
no small working set to discover.

## How it is captured — no patch, and no mode to turn off

llama.cpp names its top-k selection tensor `ffn_moe_topk-<layer>`
(`llama_context::graph_get_cb` formats every callback name that way) and
exposes `ggml_backend_sched_eval_callback` through
`llama_context_params::cb_eval`. So a routing trace is a filter on tensor
names plus a read of int32 data, through the public API.

Nothing in the llama.cpp tree is modified, and nothing is rebuilt:
`tools/trace/build.sh` links against the headers and shared objects already
there. The clean baseline SCOPE goal 6 requires is therefore the default
rather than a mode — with the tool not running, there is nothing switched on.

One shape surprised the reader and is worth knowing: llama.cpp computes the
final layer only for tokens whose logits are wanted, so in a 40-token prompt
decode layers 1–44 carry 40 rows and layer 45 carries one. Routings are
ragged, legitimately.

## What it confirms: prediction

Context beats the frequency floor by far more on real routing than the
synthetic trace suggested — **12 to 14 points, not one.**

recall@k, trace A, generated tokens only, 60 warmup tokens unscored:

| k | frequency | persistence | transition | adaptive blend |
|---|----------:|------------:|-----------:|---------------:|
| 8 | 27.6 % | 38.3 % | 36.5 % | **39.6 %** |
| 16 | 41.0 % | 49.5 % | 53.2 % | **55.1 %** |
| 32 | 59.8 % | 51.5 % | 71.8 % | **73.6 %** |

The adaptive blend moves its weight where the signal is without being told:
51 % on persistence at k=8, 57 % on transition at k=32. Frequency — the
signal the synthetic trace made look adequate — is the weakest of the three
at every k.

## What it overturns: the cache

The cache module opened by asserting that least-recently-used was the wrong
default here. On real routing that is **false**.

hit rate, trace A / trace B, generated tokens, 8.94 MB per expert:

| cache | frequency-led | recency-led | LRU |
|-------|--------------:|------------:|----:|
| 4 GB | 30.1 % / 27.2 % | 37.6 % / 35.9 % | 37.6 % / 35.9 % |
| 8 GB | 44.1 % / 40.1 % | 51.9 % / 50.9 % | 51.9 % / 50.9 % |
| 16 GB | 64.3 % / 59.8 % | 69.7 % / 68.8 % | 69.7 % / 68.8 % |

Configured as it now is by default, the policy *equals* LRU. It does not beat
it. Both traces agree, and the second was captured before any of this was
decided, precisely so the conclusion would not be fitted to the first.

Three defects had to be fixed before that table could even be produced, and
each was invisible against synthetic data:

**Recency was measured in tokens.** One token touches 360 experts, so at
token granularity almost every resident entry ties with every other. That
version reached 43.4 % against LRU's 51.9 %. Access granularity closed it.

**The heap could not track a value that changes on every access.** A stored
score goes stale the moment its entry is hit. An exact access-order front,
consulted alongside the heap, made eviction exact again — the gap was 5 to
12 points.

**The revalidation budget silently returned the wrong victim.** The docstring
said exceeding the budget "falls back to the exact scan and counts the event,
so the approximation cannot hide". The code, on exhausting the budget,
short-circuited its own condition and returned the stale candidate. It never
reached the scan, and `heap_fallbacks` read zero throughout. The claim was
true of the comment and false of the code.

## The horizon: what residency management has to work with

`benchmarks/horizon.py` asks the question the whole project turns on — how
much of the file does a window of W consecutive tokens actually need? The
widest window is reported, not the average, because a residency budget has to
survive the worst window it meets.

Trace A, with real expert sizes:

| window | experts | of model | experts GB | + floor | of file |
|-------:|--------:|---------:|-----------:|--------:|--------:|
| 1 | 360 | 6.2 % | 3.17 | **7.73** | **13.7 %** |
| 2 | 682 | 11.8 % | 6.02 | 10.57 | 18.7 % |
| 8 | 1 889 | 32.8 % | 16.67 | 21.22 | 37.6 % |
| 16 | 2 716 | 47.2 % | 23.99 | 28.54 | 50.5 % |
| 32 | 3 632 | 63.1 % | 32.09 | **36.65** | 64.9 % |
| 128 | 4 890 | 84.9 % | 43.17 | 47.73 | 84.5 % |
| 400 | 5 433 | 94.3 % | 47.93 | 52.48 | 93.0 % |

Trace B agrees to within a point at every width, and consecutive tokens share
38 % of their experts on A, 36 % on B.

Two things follow, and together they are the case for the project.

**One token needs 7.73 GB — 13.7 % of the file.** The floor is 4.55 GB of
that, so the experts a token actually routes to are 3.17 GB. A runtime that
held exactly them would run the same model in an eighth of the memory.

**A 32-token window needs 36.65 GB, and the ceiling that collapsed was
32 GB.** That is not a coincidence, it is the mechanism: llama.cpp keeps
whatever it has touched, so after thirty-odd tokens it wants more than the
ceiling allows and starts evicting things it is about to need again. The
baseline's 113.6 GB read for a 56.5 GB file is what that looks like from the
device.

**And it says where the work has to happen.** The needed set doubles by the
second token and passes half the model by the sixteenth. Anything managing
residency from *outside* the inference loop — a helper process advising the
page cache, a periodic sweep — cannot act on a horizon that short. To hold
7.73 GB instead of 56.47 GB, the decision has to be made between layers, by
something inside the loop. That is an architectural conclusion drawn from a
measurement rather than from taste, and it is what `benchmarks/residency.py`
tests the outside-the-loop alternative against.

## What is left of the value function

`value = (w_rate·rate + w_recency·recency + w_conf·confidence) × reload ÷ size`

- **rate** is the weaker signal on this model. Off by default.
- **recency** is what makes the policy equal LRU. On by default.
- **confidence**, fed from the real predictors on real routing, moved the hit
  rate by 0.1 of a point. By the time a prediction says an expert is likely
  needed, recency is already keeping it. **Prediction earns its place in the
  prefetch path — deciding what to fetch — not in eviction, deciding what to
  retain.** That is a useful thing to have learned before building goal 7's
  policy around the opposite assumption.
- **size** and **reload cost** are inert here: every expert is 9.97 MB and
  costs the same to fetch. On a mixed-quantisation model, or one tiering
  experts against attention weights, they are the whole reason for a value
  function. Untested, and honestly so.

## The general lesson, stated once

A synthetic benchmark certified a cache policy that loses to LRU on the model
it was built for, and understated the predictor's real advantage by a factor
of ten. Both errors came from the same place: the trace generator and the
thing being measured were written by the same hand, so the generator's
assumptions became the measurement's conclusions.

The generator was not useless — it caught real bugs and it is still the only
way to test without a 56 GB model. But no claim about *the model* survives it.

## Residency assist: measured, and it cannot work from outside

The horizon says a residency decision has to be made on a one- to two-token
window. `cb_eval` can act there — it fires once per MoE layer during the
forward pass, so when layer L's routing is known, layers L+1 and L+2 have not
run. That is the right place. The question is whether advising the page cache
from there does anything.

It does not, and the reason is mechanical rather than a matter of tuning.

**Fetching.** `tools/trace --horizon N` advises the page cache for the next N
layers, guessing from what they routed to for the previous token.

| arm | wall | s/token | GB read | advised |
|-----|-----:|--------:|--------:|--------:|
| no assist | 44 s | 2.76 | 88.7 | — |
| horizon 1 | 44 s | 2.75 | 89.6 | 49.5 GB |
| horizon 3 | 43 s | 2.71 | 88.9 | 144.9 GB |

144.9 GB of `WILLNEED` produced 0.3 GB of reads. The cgroup sat at its 32 GB
ceiling throughout and the kernel will not evict anything to satisfy an
advisory hint: under a full cache, `WILLNEED` is a no-op. Lead time was never
the constraint — a layer takes 61 ms here and an expert reads in about 5 —
there was simply nowhere to put it.

**Freeing.** So something has to go first. `--evict-after N` releases the
experts a layer has not routed to for N tokens; `--release-unused` also drops
the ones never routed to at all, which on a cold run is most of the file,
since llama.cpp reads all 56.5 GB during load.

| arm | wall | s/token | GB read | released | resident after |
|-----|-----:|--------:|--------:|---------:|---------------:|
| no assist | 44 s | 2.78 | 89.6 | — | 54 % |
| free after 2 tokens | 45 s | 2.83 | 90.4 | 62.8 GB | 54 % |
| free after 2, incl. never-used | 48 s | 3.01 | 90.7 | **616.9 GB** | 54 % |
| fetch 2 and free 2, incl. never-used | 51 s | 3.17 | 88.9 | 616.9 GB + 97.8 advised | 54 % |

**616.9 GB of `DONTNEED`, across 209 937 calls, and residency did not move by
a percentage point.** Throughput gets steadily worse as more advice is issued,
which is the cost of issuing it. Every call returned success.

**Why, in isolation.** `posix_fadvise(DONTNEED)` drops clean page-cache pages
— but not ones a live process holds mapped, because the mapping keeps a
reference:

| | resident |
|---|---:|
| freshly written, `fsync` then `DONTNEED` | 0.0 % |
| after another process mmaps it and touches every page | 100.0 % |
| **`DONTNEED` from a second fd while that process holds it** | **100.0 %** |
| `DONTNEED` after that process exits | 0.0 % |

The call succeeds. It simply does nothing. `tests/test_bench.py` carries this
as a test, so a platform where it behaves differently fails loudly rather than
quietly invalidating the conclusion below.

### What that settles

**Advisory page-cache management cannot manage a mmap'd model, in either
direction.** `WILLNEED` has nowhere to read into when the ceiling is full;
`DONTNEED` cannot release what the mapping is holding. Between them there is
no way to shape residency from beside the runtime.

It also explains the one thing that *did* help. The floor pinner bought
12.5 % — and it worked by **reading**, not by advising. Reads move pages;
advice does not.

So TierInfer cannot assist llama.cpp's residency. It has to own the loading
path — explicit `pread` into buffers it controls, with no mapping in the way.
Which is what `tierinfer.storage` and `tierinfer.stream` already are, measured
at 3.47× demand paging and 3.0 GB/s against llama.cpp's own 1.54.

That is goal 6's real finding: the integration cannot be advisory, and the
measurements say so three separate ways.

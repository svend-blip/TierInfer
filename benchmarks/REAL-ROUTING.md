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

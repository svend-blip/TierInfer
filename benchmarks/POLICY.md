# One policy over three tiers

Measured 2026-09-17 against the real routing traces, with the per-tier costs
measured elsewhere in this project:

| from | seconds | source |
|------|--------:|--------|
| VRAM | 0 | it is already there |
| RAM → VRAM | 0.36 ms | pinned, 27.6 GB/s, `VRAM.md` |
| NVMe → RAM → VRAM | 3.66 ms | 3.0 GB/s, `STREAMING.md` |

Ten times apart. That ratio is the shape of everything below.

```
python benchmarks/policy.py MODEL.gguf TRACE.jsonl --context 16384
```

VRAM sized by the goal-9 budget (2 297 experts, 22.4 GB at 16 k context),
RAM 64 GB, model 56.5 GB, 400 generated tokens of 360 activations each.

| policy | served without NVMe | VRAM hits | NVMe reads/token | prefetch used | ms/token | of a warm token |
|--------|--------------------:|----------:|-----------------:|--------------:|---------:|----------------:|
| nvme-only | 0.0 % | 0.0 % | 360 | — | 1 317.6 | 777 % |
| ram-only | 96.2 % | 0.0 % | 14 | — | 174.4 | 103 % |
| fixed-0 | 96.2 % | 76.4 % | 14 | — | 75.4 | 44 % |
| fixed-8 | 96.2 % | 76.7 % | 14 | 64 % | 75.0 | 44 % |
| fixed-16 | 96.2 % | 77.4 % | 14 | 41 % | 74.7 | 44 % |
| **adaptive** | 96.2 % | 77.4 % | 14 | 44 % | **74.7** | 44 % |

## What it says

**Tiering is worth 17×; the dial on top of it is worth 1 %.** Between
nvme-only and fixed-0 the cost falls from 1 317 ms to 75. Between fixed-0 and
the best speculation it falls from 75.4 to 74.7. Those two facts belong in the
same table so that the second is not mistaken for the first.

**The adaptive arm reaches the best fixed arm without being told which it is.**
It starts at depth 8, moves twelve times on measured stalls against measured
waste, settles near 20, and lands on the same 74.7 ms as the best depth found
by trying them all. That is what the mechanism is for: not beating a tuned
constant, but not needing one.

**A 3.8 % miss rate costs two thirds of the time.** 14 of 360 activations per
token reach NVMe, and at 3.66 ms each that is roughly 50 of the 75 ms. At a
tenfold cost ratio the tail dominates long before it looks like it should,
which is the argument for spending effort on the last few percent of
residency rather than on the first eighty.

**Prefetching cannot reach that tail.** The experts that miss are the ones no
predictor ranked in its top-k — raising the depth from 8 to 16 lifted VRAM
hits by 0.7 of a point and left NVMe reads at 14. Speculation moves experts
between the two cheap tiers; it does not find the ones nobody saw coming.

## What this is and is not

It is a cost model over measured constants, applied to real routing. It
answers "how much transfer time does this policy incur", which is the part a
policy controls.

It is not an inference run, and the ms/token figures are not throughput. No
compute overlaps the transfers here, so every number is a ceiling on the cost
rather than a prediction of the wall clock.

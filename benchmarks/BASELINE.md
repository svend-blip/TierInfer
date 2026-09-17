# Baseline — what a 56 GB model costs when the memory is not there

Measured 2026-09-17 on this host. Every figure comes from
`benchmarks/baseline.py`, which is reproducible:

```
python benchmarks/baseline.py \
    /home/svend/models/GLM-4.5-Air-Derestricted-IQ4_XS/GLM-4.5-Air-Derestricted.IQ4_XS.gguf \
    --tokens 8 --threads 16 --limit 32 --limit 16 --budget 1200
```

**Host** 187 GB RAM, NVMe (`nvme0n1p2`, ext4), 16 threads, llama.cpp
`b9888-cb295bf59`, CPU only (`-ngl 0`).
**Model** GLM-4.5-Air-Derestricted IQ4_XS, 56.47 GB, `glm4moe`: 47 layers,
46 of them MoE, 128 experts each, 8 routed per token.

| condition | gen t/s | prompt t/s | wall | GB read | GB/s | IOPS | mean read | await | resident after | peak |
|-----------|--------:|-----------:|-----:|--------:|-----:|-----:|----------:|------:|---------------:|-----:|
| warm      |    6.00 |         18 |  17 s |     0.0 |    — |    2 |       4 KB | 0.54 ms |          100 % |    — |
| cold      |    5.90 |         15 |  37 s |    56.5 | 1.54 | 12,642 |    128 KB | 0.14 ms |          100 % |    — |
| cold + 32 GB |  0.80 |          1 |  68 s |   113.6 | 1.67 | 39,916 |     44 KB | 0.15 ms |           14 % | 32.0 GB |
| cold + 16 GB |     — |          — |  39 s |    56.5 | 1.45 | 11,884 |    128 KB | 0.13 ms |            0 % | 16.0 GB |

The 16 GB run has no throughput because there was none: the cgroup OOM killer
stopped it (`exit -9`) while the model was still loading. The ceiling is below
what llama.cpp needs to start at all.

## What the numbers say

**Cold costs 20 seconds and buys nothing back.** 56.5 GB moves at 1.54 GB/s in
128 KB reads — near this device's sequential rate, and readahead is doing its
job. Once loaded, generation is identical to warm (5.90 against 6.00): the
storage path stops mattering the moment the weights are resident.

**Eight tokens read the entire model.** Eight tokens route to at most
8 × 46 = 368 expert slabs, about 3.2 GB of the 56.5 GB file. All 56.5 GB was
read anyway. That gap is the whole premise of this project stated as a
measurement: the runtime moves the model, not the experts.

**A ceiling below the model is not a slowdown, it is a collapse.** At 32 GB —
57 % of the model, a generous ratio — generation falls from 5.90 to 0.80 t/s,
7.4× slower. Three things happen together and each confirms the others:

- **113.6 GB is read for a 56.5 GB model.** Every byte moves twice on
  average. Pages are evicted before they are used again and fetched back.
- **Mean read size falls from 128 KB to 44 KB** while IOPS triples to 39,916.
  Thrashing does not just read more, it reads worse: the pattern degrades
  from readahead-friendly streaming into scattered demand faults.
- **14 % of the model is resident at the end** of a run that read twice its
  size, so the page cache is holding almost nothing it fetched.

Bandwidth is *not* the constraint — 1.67 GB/s under the ceiling is slightly
higher than the 1.54 GB/s cold. The device is keeping up. What collapses is
which bytes are chosen and how many times each is fetched, which is a policy
problem, and policy is what TierInfer proposes to supply.

## What this measurement cannot say

**Mean read size is a mean, not a distribution.** 44 KB is consistent with a
uniform 44 KB stream and with a mix of 4 KB faults and 2 MB readahead, and
those are different worlds. A histogram needs `blktrace`, which needs root.

**CPU-only.** `-ngl 0` keeps the GPU out so the storage path is what varies.
Absolute throughput here is not what this host does with GPU offload, and the
7.4× ratio is between two CPU-only runs, not a claim about a production
configuration.

**One prompt, one model, one device.** Nothing here establishes that the
collapse ratio generalises. It establishes that it is real on the machine
TierInfer is being built for.

## How each number was obtained

- **Throughput** from the runtime's own report, parsed from
  `[ Prompt: X t/s | Generation: Y t/s ]`.
- **Cold** via `posix_fadvise(DONTNEED)` after an `fsync`, verified by
  `mincore` before the run — dirty pages are not dropped, so a file that was
  just written stays resident while reporting a successful eviction.
- **Ceiling** via `systemd-run --user --scope -p MemoryMax=`, with the peak
  sampled from the scope's own `memory.peak` while it runs, because systemd
  removes a transient scope before it can be asked afterwards.
- **Device figures** from `/proc/diskstats` deltas on the partition, not the
  whole disk, so traffic to other filesystems stays out.
- **Residency** from `mincore` over the model file.

None of it needs root.

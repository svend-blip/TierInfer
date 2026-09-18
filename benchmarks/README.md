# Benchmarks

Each script here measures one claim and each `.md` beside it records what
came out, with the command that produced it. None of the numbers is a plan.

| file | question | result |
|---|---|---|
| `baseline.py` → `BASELINE.md` | what does a 32 GB ceiling cost llama.cpp on a 56 GB model? | 7.4× throughput; 113.6 GB read for a 56.5 GB file |
| `streaming.py` → `STREAMING.md` | can explicit reads beat demand paging? | 3.47× at eight workers; a single-threaded `pread` is *slower* than faulting |
| `read_paths.py` | how much of the paging cost is asking 4 KB at a time? | 4× the time, 763× the operations |
| `cache_policy.py`, `prediction.py` | synthetic traces — superseded by real routing | see `REAL-ROUTING.md` for what the synthetic numbers got wrong |
| `horizon.py` → `REAL-ROUTING.md` | how much of the file does a window of W tokens need? | 13.7 % for one token, 64.9 % for 32 |
| `residency.py` → `residency-report.json` | can a helper pin the floor / advise the page cache? | reading helps 12.5 %; advice does nothing |
| `vram.py` → `VRAM.md` | what fits on the card, and at what transfer rate? | 22–25 GB of experts at 76–79 % hit rate; pinned 27.6 GB/s |
| `policy.py` → `POLICY.md` | a **cost simulation** of one policy over three tiers | tiering 17×, the dial 1 % — modelled, not run |

The `inloop.py` in-loop advisory experiment and the trace tool's assist mode
were removed on 2026-09-18 after the audit: the measurement that falsified
them stays in `REAL-ROUTING.md` and `inloop-report.json`.

The TierInfer-enabled condition of goal 2 does not exist yet, because no
loader does (`SCOPE.md` goal 6).

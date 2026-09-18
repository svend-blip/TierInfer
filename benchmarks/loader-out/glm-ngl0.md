# glm-ngl0: native vs loader (3 runs each, n_predict 32, extra -ngl 0 -fa off)

| arm | run | load s | prompt t/s | gen t/s | infer GB | infer reads | mean KB | await ms | RSS GB | tokens identical to native #1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| native | 1 | 30 | 4.320 | 0.310 | 61.21 | 810079 | 79 | 802.33 | 32 | yes |
| loader | 1 | 23 | 2.328 | 0.324 | 68.48 | 572497 | 125 | 127.12 | 28 | yes |
| native | 2 | 36 | 4.043 | 0.107 | 67.37 | 907126 | 78 | 753.76 | 32 | yes |
| loader | 2 | 175 | 22.640 | 0.140 | 0.01 | 295 | 35 | 0.68 | 57 | yes |
| native | 3 | 35 | 3.349 | 0.325 | 70.75 | 857324 | 87 | -4450.53 | 32 | yes |
| loader | 3 | 23 | 31.643 | 2.507 | 0.00 | 5 | 18 | 0.80 | 57 | yes |
| **native median** | | | | **0.310** (0.107–0.325) | 67.37 | 857324 | | | | |
| **loader median** | | | | **0.324** (0.140–2.507) | 0.01 | 295 | | | | |

| loader run | hits | misses | hit rate | faults | copied GB | evictions | prefetch issued/useful/late/wasted |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | 7349 | 7865 | 48.3% | 136380 | 75.1 | 5435 | 0/0/50/0 |
| 2 | 0 | 0 | 0.0% | 0 | 0.0 | 0 | 0/0/0/0 |
| 3 | 0 | 0 | 0.0% | 0 | 0.0 | 0 | 0/0/0/0 |

Greedy tokens identical across all runs and arms: **True**

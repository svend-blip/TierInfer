# glm-d8-prerouter: native vs loader (2 runs each, n_predict 32, extra -ngl 0 -fa off)

| arm | run | load s | prompt t/s | gen t/s | infer GB | infer reads | mean KB | await ms | RSS GB | tokens identical to native #1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| loader | 1 | 21 | 2.681 | 0.193 | 72.08 | 202196 | 374 | 1.93 | 28 | — |
| loader | 2 | 20 | 2.668 | 0.194 | 72.10 | 202811 | 373 | 1.92 | 28 | — |
| **loader median** | | | | **0.194** (0.193–0.194) | 72.09 | 202504 | | | | |

| loader run | hits | misses | hit rate | faults | copied GB | evictions | prefetch issued/useful/late/wasted |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | 8769 | 7525 | 53.8% | 129125 | 86.4 | 6716 | 2110/818/148/726 |
| 2 | 8767 | 7527 | 53.8% | 129313 | 86.4 | 6718 | 2112/817/150/728 |



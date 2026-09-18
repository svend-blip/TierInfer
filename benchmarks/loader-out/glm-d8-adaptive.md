# glm-d8-adaptive: native vs loader (2 runs each, n_predict 32, extra -ngl 0 -fa off)

| arm | run | load s | prompt t/s | gen t/s | infer GB | infer reads | mean KB | await ms | RSS GB | tokens identical to native #1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| loader | 1 | 19 | 2.528 | 0.340 | 70.50 | 197493 | 374 | 1.81 | 28 | — |
| loader | 2 | 19 | 2.597 | 0.355 | 70.50 | 197173 | 375 | 1.82 | 28 | — |
| **loader median** | | | | **0.347** (0.340–0.355) | 70.50 | 197333 | | | | |

| loader run | hits | misses | hit rate | faults | copied GB | evictions | prefetch issued/useful/late/wasted |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | 8406 | 7888 | 51.6% | 132473 | 79.4 | 5914 | 411/304/140/98 |
| 2 | 8406 | 7888 | 51.6% | 132481 | 79.4 | 5914 | 413/304/129/98 |



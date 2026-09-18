# q480b-ncmoe60: native vs loader (-t 32 --no-warmup -ngl 99 -ncmoe 60 -fa on)

| arm | run | load s | prompt t/s | gen t/s | infer GiB | infer reads | mean KB | await ms | tokens = native #1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| native | 1 | 193 | 0.774 | 0.437 | 248.0 | 2440559 | 107 | 4.9 | yes |
| loader | 1 | 37 | 0.750 | 0.622 | 157.9 | 427924 | 387 | 2.0 | yes |
| native | 2 | 192 | 0.765 | 0.481 | 246.1 | 2255392 | 114 | 4.8 | yes |
| loader | 2 | 37 | 0.751 | 0.668 | 157.9 | 426373 | 388 | 2.0 | yes |
| native | 3 | 189 | 0.767 | 0.467 | 244.7 | 2207410 | 116 | 4.8 | yes |
| loader | 3 | 37 | 0.749 | 0.647 | 157.9 | 426882 | 388 | 2.0 | yes |
| **native median** | 3 | 192 | 0.767 | **0.467** (0.437–0.481) | 246.1 | 2255392 | | | |
| **loader median** | 3 | 37 | 0.750 | **0.647** (0.622–0.668) | 157.9 | 426882 | | | |

| loader run | hit rate (gen, median) | faults/token | MB copied/token | evictions/token | wall ms/token | run: faults | resident-page faults | repaired | evictions | UNMAPs | forgotten |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 93.6% | 832 | 454 | 16 | 1447 | 59212 | 47502 | 387 | 885 | 28 | 5851 |
| 2 | 93.6% | 832 | 454 | 16 | 1314 | 59201 | 47700 | 378 | 885 | 28 | 5851 |
| 3 | 93.6% | 832 | 454 | 16 | 1309 | 59207 | 47787 | 396 | 885 | 28 | 5851 |

Greedy tokens identical across all runs and arms: **True**

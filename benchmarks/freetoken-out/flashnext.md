# flashnext: FreeToken native vs TierInfer-tiered CPU-executor layers

| run | arm | tier GB | depth | load s | wall s | FT decode t/s | infer GiB | reads | mean KB | load GiB | output = first |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| flashnext-fix-t8-tiered-1 | tiered | 8.0 | 0 | 44 | 38.5 | 6.1 | 18.52 | 82640 | 235 | 57.3 | yes |
| flashnext-fix-t8d8-tiered-1 | tiered | 8.0 | 0 | 44 | 38.2 | 6.3 | 18.63 | 83163 | 235 | 57.3 | yes |
| flashnext-fix-t8d8-tiered-2 | tiered | 8.0 | 0 | 44 | 38.8 | 6.2 | 18.63 | 83158 | 235 | 57.3 | yes |
| flashnext-native-1 | native | — | — | 55 | 4.3 | 34.8 | 0.01 | 2472 | 4 | 72.7 | yes |
| flashnext-native-2 | native | — | — | 54 | 4.3 | 35.0 | 0.01 | 2472 | 4 | 72.7 | yes |
| flashnext-t16-tiered-1 | tiered | 16.0 | 0 | 44 | 25.1 | 35.4 | 15.46 | 69158 | 234 | 57.3 | yes |
| flashnext-t16-tiered-2 | tiered | 16.0 | 0 | 44 | 25.2 | 34.8 | 15.46 | 69214 | 234 | 57.3 | yes |
| flashnext-t8d8-tiered-1 | tiered | 8.0 | 0 | 44 | 39.6 | 5.4 | 18.73 | 83580 | 235 | 57.3 | yes |
| flashnext-t8d8-tiered-2 | tiered | 8.0 | 0 | 44 | 40.2 | 5.4 | 18.73 | 83628 | 235 | 57.3 | yes |
| flashnext-tiered-1 | tiered | 8.0 | 0 | 45 | 40.3 | 5.5 | 18.74 | 83767 | 235 | 57.3 | yes |
| flashnext-tiered-2 | tiered | 8.0 | 0 | 44 | 39.6 | 5.4 | 18.73 | 83660 | 235 | 57.3 | yes |

| run | prefill GB through tier | prefill faults | hit rate (decode) | MB/token | faults/token | evictions/token | ms/token | prefetch issued/useful/late/wasted |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| flashnext-fix-t8-tiered-1 | 17.3 | 9228 | 85.8% | 42 | 644 | 15 | 229 | 0/0/0/0 |
| flashnext-fix-t8d8-tiered-1 | 17.3 | 9220 | 85.8% | 42 | 645 | 15 | 212 | 48/14/0/0 |
| flashnext-fix-t8d8-tiered-2 | 17.3 | 9269 | 85.8% | 42 | 705 | 15 | 226 | 48/14/0/0 |
| flashnext-t16-tiered-1 | 17.1 | 9153 | 100.0% | 0 | 0 | 0 | 27 | 0/0/0/0 |
| flashnext-t16-tiered-2 | 17.1 | 9151 | 100.0% | 0 | 0 | 0 | 27 | 0/0/0/0 |
| flashnext-t8d8-tiered-1 | 17.3 | 9285 | 100.0% | 50 | 750 | 18 | 241 | 0/0/0/0 |
| flashnext-t8d8-tiered-2 | 17.3 | 9257 | 100.0% | 50 | 735 | 18 | 245 | 0/0/0/0 |
| flashnext-tiered-1 | 17.3 | 9209 | 100.0% | 50 | 796 | 18 | 246 | 0/0/0/0 |
| flashnext-tiered-2 | 17.3 | 9218 | 100.0% | 50 | 764 | 18 | 246 | 0/0/0/0 |

Greedy output identical across all runs and arms: **True**

#!/usr/bin/env bash
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b
$S/glm_crosscheck.sh > $S/chain1.log 2>&1
# cold: drop the six shards, then native -ngl 0, no warmup, 32 tokens
cd ~/TierInfer && PYTHONPATH=src python3 -c "from tierinfer.bench import drop_cache; r=drop_cache('/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf'); print('cold residency', r.fraction)" >> $S/chain1.log 2>&1
$S/serve_run.sh native-ngl0-cold1 $S/base 32 -- -m /data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf -ngl 0 -c 4096 -t 32 --no-warmup -fa off >> $S/chain1.log 2>&1
echo chain1-done >> $S/chain1.log

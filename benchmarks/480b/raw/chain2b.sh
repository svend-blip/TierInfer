#!/usr/bin/env bash
# replaces chain2 after the orphaned-server incident: cold2, warm2, cold3, warm3, then the GLM crosscheck
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf; cd ~/TierInfer
run() { $S/serve_run.sh "$1" $S/base 32 -- -m $M1 -ngl 0 -c 4096 -t 32 --no-warmup -fa off >> $S/chain2.log 2>&1; }
drop() { PYTHONPATH=src python3 -c "from tierinfer.bench import drop_cache; r=drop_cache('$M1'); print('cold residency', r.fraction)" >> $S/chain2.log 2>&1; }
echo "--- chain2b start $(date -u +%FT%TZ): chain2 stopped after warm1 (its server was cold1's orphan; warm1 is valid, cold2 discarded)" >> $S/chain2.log
drop; run native-ngl0-cold2
run native-ngl0-warm2
drop; run native-ngl0-cold3
run native-ngl0-warm3
$S/glm_crosscheck.sh >> $S/chain2.log 2>&1
echo chain2-done >> $S/chain2.log

#!/usr/bin/env bash
# after chain3: GPU offload sweep, native llama-server. all attention on GPU first, then experts of the last layers.
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf
until grep -q chain3-done $S/chain3.log 2>/dev/null; do sleep 20; done
cd ~/TierInfer
drop() { PYTHONPATH=src python3 -c "from tierinfer.bench import drop_cache; r=drop_cache('$M1'); print('cold residency', r.fraction)" >> $S/chain4.log 2>&1; }
for ncmoe in 62 60 58 57 56; do
  drop
  $S/serve_run.sh native-ngl99-ncmoe$ncmoe-cold $S/sweep 32 -- -m $M1 -ngl 99 -ncmoe $ncmoe -c 4096 -t 32 --no-warmup -fa on >> $S/chain4.log 2>&1
  rc=$(cat $S/sweep/native-ngl99-ncmoe$ncmoe-cold.rc 2>/dev/null || echo 1)
  echo "ncmoe $ncmoe rc=$rc" >> $S/chain4.log
  if [ "$rc" != 0 ]; then echo "stopping sweep at ncmoe $ncmoe" >> $S/chain4.log; break; fi
  $S/serve_run.sh native-ngl99-ncmoe$ncmoe-warm $S/sweep 32 -- -m $M1 -ngl 99 -ncmoe $ncmoe -c 4096 -t 32 --no-warmup -fa on >> $S/chain4.log 2>&1
done
echo chain4-done >> $S/chain4.log

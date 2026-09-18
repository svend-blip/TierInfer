#!/usr/bin/env bash
# after chain2: capture two 400-token routing traces from the 480B (attention on GPU, experts on CPU)
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf
until grep -q chain2-done $S/chain2.log 2>/dev/null; do sleep 20; done
cd ~/TierInfer; mkdir -p traces
for arm in prose code; do
  f=$S/prompt.txt; [ $arm = code ] && f=$S/prompt-code.txt
  python3 $S/sample.py $S/trace-$arm.samples.csv 2 md0 sda sdb & SP=$!
  grep -E " (md0|sda|sdb) " /proc/diskstats > $S/trace-$arm.diskstats.before
  /usr/bin/time -v -o $S/trace-$arm.time ./build/tierinfer-trace -m $M1 -ngl 99 --cpu-moe -t 32 -c 4096 -n 400 -f $f -o traces/qwen3coder480b-$arm-400.jsonl 2> $S/trace-$arm.err
  echo "trace $arm rc=$?" >> $S/chain3.log
  grep -E " (md0|sda|sdb) " /proc/diskstats > $S/trace-$arm.diskstats.after
  kill $SP; wait $SP 2>/dev/null
done
echo chain3-done >> $S/chain3.log

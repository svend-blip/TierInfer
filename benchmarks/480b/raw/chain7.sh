#!/usr/bin/env bash
# separates the two effects seen in b-cache-d0-100g: reload-cost noise (fixed) and demand batching (now a switch)
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf; cd ~/TierInfer; T=traces/qwen3coder480b-prose-400.jsonl; TC=traces/qwen3coder480b-code-400.jsonl; OUT=benchmarks/replay-out/480b
rp() { local label=$1 memmax=$2; shift 2
  python3 $S/sample.py $OUT/$label.samples.csv 2 md0 sda sdb & local SP=$!
  systemd-run --user --scope --quiet -p MemoryMax=$memmax -p MemorySwapMax=0 env PYTHONPATH=src python3 benchmarks/replay.py $M1 $T --out $OUT --label $label "$@" > $OUT/$label.log 2>&1
  echo "$label rc=$?" >> $S/chain5.log; kill $SP; wait $SP 2>/dev/null; }
M="--predictor-warmup 50 --tokens 150 --drop-after-read --cold"
rp c-cache-d0-100g-serial  140G --depth 0 --ram-gb 100 $M --no-batch-demand
rp c-cache-d0-100g-batched 140G --depth 0 --ram-gb 100 $M
rp c-cache-d0-150g-serial  178G --depth 0 --ram-gb 150 $M --no-batch-demand
rp c-pf-d8-100g-serial     140G --depth 8 --ram-gb 100 $M --no-batch-demand
T=$TC rp c-cache-d0-100g-code-serial 140G --depth 0 --ram-gb 100 $M --no-batch-demand
echo chain7-done >> $S/chain5.log

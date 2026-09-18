#!/usr/bin/env bash
# after chain5d: the same arms with demand reads batched through the streamer
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf; cd ~/TierInfer; T=traces/qwen3coder480b-prose-400.jsonl; TC=traces/qwen3coder480b-code-400.jsonl; OUT=benchmarks/replay-out/480b
until grep -q chain5d-done $S/chain5.log 2>/dev/null; do sleep 20; done
rp() { local label=$1 memmax=$2; shift 2
  python3 $S/sample.py $OUT/$label.samples.csv 2 md0 sda sdb & local SP=$!
  systemd-run --user --scope --quiet -p MemoryMax=$memmax -p MemorySwapMax=0 env PYTHONPATH=src python3 benchmarks/replay.py $M1 $T --out $OUT --label $label "$@" > $OUT/$label.log 2>&1
  echo "$label rc=$?" >> $S/chain5.log; kill $SP; wait $SP 2>/dev/null; }
M="--predictor-warmup 50 --tokens 150 --drop-after-read --cold"
rp b-verify 140G --depth 0 --ram-gb 60 --predictor-warmup 30 --tokens 3 --verify-every 1 --drop-after-read --cold
rp b-cache-d0-100g 140G --depth 0 --ram-gb 100 $M
rp b-pf-d8-100g    140G --depth 8 --ram-gb 100 $M
rp b-pf-d8-100g-comp 140G --depth 8 --ram-gb 100 $M --attn-ms 4 --ffn-ms 4
rp b-cache-d0-150g 178G --depth 0 --ram-gb 150 $M
T=$TC rp b-cache-d0-100g-code 140G --depth 0 --ram-gb 100 $M
echo chain6-done >> $S/chain5.log

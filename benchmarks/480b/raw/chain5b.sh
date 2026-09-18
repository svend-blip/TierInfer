#!/usr/bin/env bash
# replaces the rest of chain5: injections with a warmed predictor, then the measurement arms from token 50
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf; cd ~/TierInfer
T=traces/qwen3coder480b-prose-400.jsonl; TC=traces/qwen3coder480b-code-400.jsonl
OUT=benchmarks/replay-out/480b; mkdir -p $OUT
# wait for whatever replay arm chain5 left running
while ps -eo args | grep -q "[b]enchmarks/replay.py"; do sleep 10; done
echo "--- chain5b start $(date -u +%FT%TZ)" >> $S/chain5.log
rp() { local label=$1 memmax=$2; shift 2
  python3 $S/sample.py $OUT/$label.samples.csv 2 md0 sda sdb & local SP=$!
  systemd-run --user --scope --quiet -p MemoryMax=$memmax -p MemorySwapMax=0 \
    env PYTHONPATH=src python3 benchmarks/replay.py $M1 $T --out $OUT --label $label "$@" > $OUT/$label.log 2>&1
  echo "$label rc=$?" >> $S/chain5.log
  kill $SP; wait $SP 2>/dev/null; }
W="--predictor-warmup 30 --tokens 8 --verify-every 1 --drop-after-read --ram-gb 60"
rp inj2-fail-reads    140G --depth 8 $W --inject fail-reads
rp inj2-tiny-pool     140G --depth 8 $W --inject tiny-pool
rp inj2-tiny-vram     140G --depth 8 $W --inject tiny-vram
rp inj2-missing-range 140G --depth 8 $W --inject missing-range
M="--predictor-warmup 50 --tokens 150 --drop-after-read --cold"
rp cache-d0-100g   140G --depth 0  --ram-gb 100 $M
rp pf-d8-100g      140G --depth 8  --ram-gb 100 $M
rp pf-d16-100g     140G --depth 16 --ram-gb 100 $M
rp pf-d8-100g-comp 140G --depth 8  --ram-gb 100 $M --attn-ms 4 --ffn-ms 4
rp pf-d8-100g-vram 140G --depth 8  --ram-gb 100 $M --vram
rp pf-d8-150g      178G --depth 8  --ram-gb 150 $M
T=$TC rp pf-d8-100g-code 140G --depth 8 --ram-gb 100 $M
rp pf-d8-100g-pagecache 178G --depth 8 --ram-gb 100 --predictor-warmup 50 --tokens 150 --cold
echo chain5-done >> $S/chain5.log

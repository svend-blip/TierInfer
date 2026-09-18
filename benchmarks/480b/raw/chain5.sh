#!/usr/bin/env bash
# after chain4: TierInfer replay arms on the captured 480B routing. Real I/O, no compute.
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf; cd ~/TierInfer
T=traces/qwen3coder480b-prose-400.jsonl; TC=traces/qwen3coder480b-code-400.jsonl
OUT=benchmarks/replay-out/480b; mkdir -p $OUT
until grep -q chain4-done $S/chain4.log 2>/dev/null; do sleep 20; done
[ -s $T ] || { echo "no trace" >> $S/chain5.log; exit 1; }
rp() { # label memmax args...
  local label=$1 memmax=$2; shift 2
  python3 $S/sample.py $OUT/$label.samples.csv 2 md0 sda sdb & local SP=$!
  systemd-run --user --scope --quiet -p MemoryMax=$memmax -p MemorySwapMax=0 \
    env PYTHONPATH=src python3 benchmarks/replay.py $M1 $T --out $OUT --label $label "$@" > $OUT/$label.log 2>&1
  echo "$label rc=$?" >> $S/chain5.log
  kill $SP; wait $SP 2>/dev/null
}
# 0. correctness first: every delivery verified against an exact read, prefetch on
rp verify-d8 140G --depth 8 --ram-gb 100 --tokens 3 --verify-every 1 --drop-after-read --cold
# 1. failure injection, short, every delivery verified
rp inj-bad-predictor 140G --depth 8 --ram-gb 60 --tokens 8 --verify-every 1 --inject bad-predictor --drop-after-read
rp inj-fail-reads    140G --depth 8 --ram-gb 60 --tokens 8 --verify-every 1 --inject fail-reads --drop-after-read
rp inj-tiny-pool     140G --depth 8 --ram-gb 60 --tokens 8 --verify-every 1 --inject tiny-pool --drop-after-read
rp inj-tiny-vram     140G --depth 8 --ram-gb 60 --tokens 8 --verify-every 1 --inject tiny-vram --drop-after-read
rp inj-missing-range 140G --depth 8 --ram-gb 60 --tokens 8 --verify-every 1 --inject missing-range --drop-after-read
# 2. the measurement arms: TierInfer's cache is the RAM tier (page cache dropped after each read), cold start
rp cache-d0-100g   140G --depth 0  --ram-gb 100 --tokens 150 --drop-after-read --cold
rp pf-d8-100g      140G --depth 8  --ram-gb 100 --tokens 150 --drop-after-read --cold
rp pf-d16-100g     140G --depth 16 --ram-gb 100 --tokens 150 --drop-after-read --cold
rp pf-d8-100g-comp 140G --depth 8  --ram-gb 100 --tokens 150 --drop-after-read --cold --attn-ms 4 --ffn-ms 4
rp pf-d8-100g-vram 140G --depth 8  --ram-gb 100 --tokens 150 --drop-after-read --cold --vram
rp pf-d8-150g      178G --depth 8  --ram-gb 150 --tokens 150 --drop-after-read --cold
# 3. same treatment, other prompt class
T=$TC rp pf-d8-100g-code 140G --depth 8 --ram-gb 100 --tokens 150 --drop-after-read --cold
# 4. page cache allowed to help (what a naive deployment would see)
rp pf-d8-100g-pagecache 178G --depth 8 --ram-gb 100 --tokens 150 --cold
echo chain5-done >> $S/chain5.log

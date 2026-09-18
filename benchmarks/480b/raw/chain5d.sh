#!/usr/bin/env bash
set -uo pipefail
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b; M1=/data/ai-data/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf; cd ~/TierInfer; T=traces/qwen3coder480b-prose-400.jsonl; OUT=benchmarks/replay-out/480b
until grep -q chain5c-done $S/chain5.log 2>/dev/null; do sleep 20; done
systemd-run --user --scope --quiet -p MemoryMax=140G -p MemorySwapMax=0 env PYTHONPATH=src python3 benchmarks/replay.py $M1 $T --out $OUT --label inj3-tiny-vram --depth 8 --predictor-warmup 30 --tokens 8 --verify-every 1 --drop-after-read --ram-gb 60 --inject tiny-vram > $OUT/inj3-tiny-vram.log 2>&1
echo "inj3-tiny-vram rc=$?" >> $S/chain5.log
echo chain5d-done >> $S/chain5.log

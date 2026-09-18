#!/usr/bin/env bash
# Batched prompt decode vs one token per decode on GLM-4.5-Air: the routing must agree.
set -euo pipefail
S=$(dirname "$0"); M=$HOME/models/GLM-4.5-Air-Derestricted-IQ4_XS/GLM-4.5-Air-Derestricted.IQ4_XS.gguf
P="Write a detailed technical explanation of how mixture-of-experts routing works in modern transformer language models, covering the router, top-k selection and load balancing."
cd ~/TierInfer
./build/tierinfer-trace -m "$M" -ngl 0 -t 32 -c 1024 -n 2 -p "$P" -o "$S/glm-batched.jsonl" 2> "$S/glm-batched.err"
./build/tierinfer-trace -m "$M" -ngl 0 -t 32 -c 1024 -n 2 --one-by-one -p "$P" -o "$S/glm-onebyone.jsonl" 2> "$S/glm-onebyone.err"
echo crosscheck-done

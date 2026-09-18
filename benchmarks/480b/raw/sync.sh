#!/usr/bin/env bash
# copy run artefacts from the session scratchpad into the repository (raw evidence, committed)
set -euo pipefail
S=$(dirname "$0"); D=$HOME/TierInfer/benchmarks/480b/raw; mkdir -p "$D"
rsync -a --exclude '*.server.stdout' "$S"/base "$S"/sweep "$S"/out "$D"/ 2>/dev/null || true
cp -f "$S"/*.log "$S"/*.err "$S"/*.csv "$S"/*.txt "$S"/*.sh "$S"/*.py "$D"/ 2>/dev/null || true
du -sh "$D"

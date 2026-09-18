#!/usr/bin/env bash
# run_llama.sh LABEL OUTDIR -- <llama-cli args>   ; records counters, RSS, wall, stdout
set -euo pipefail
LABEL=$1; OUT=$2; shift 2; [ "$1" = "--" ] && shift
mkdir -p "$OUT"
S=$(dirname "$0")
python3 "$S/sample.py" "$OUT/$LABEL.samples.csv" 2 md0 sda sdb & SP=$!
grep -E " (md0|sda|sdb) " /proc/diskstats > "$OUT/$LABEL.diskstats.before"
grep -E "MemAvailable|Cached:|Shmem:" /proc/meminfo > "$OUT/$LABEL.meminfo.before"
date -u +%FT%TZ > "$OUT/$LABEL.start"
set +e
/usr/bin/time -v -o "$OUT/$LABEL.time" "$@" < /dev/null > "$OUT/$LABEL.stdout" 2> "$OUT/$LABEL.stderr"
RC=$?
set -e
date -u +%FT%TZ > "$OUT/$LABEL.end"
echo "$RC" > "$OUT/$LABEL.rc"
grep -E " (md0|sda|sdb) " /proc/diskstats > "$OUT/$LABEL.diskstats.after"
grep -E "MemAvailable|Cached:|Shmem:" /proc/meminfo > "$OUT/$LABEL.meminfo.after"
kill $SP 2>/dev/null; wait $SP 2>/dev/null || true
echo "$LABEL rc=$RC"

#!/usr/bin/env bash
# serve_run.sh LABEL OUTDIR N_PREDICT -- <llama-server args>
# Starts llama-server, waits for /health, times one /completion with the fixed prompt, stops it.
set -euo pipefail
LABEL=$1; OUT=$2; NPRED=$3; shift 3; [ "$1" = "--" ] && shift
S=$(dirname "$0"); mkdir -p "$OUT"; PORT=8931
python3 "$S/sample.py" "$OUT/$LABEL.samples.csv" 2 md0 sda sdb & SP=$!
grep -E " (md0|sda|sdb) " /proc/diskstats > "$OUT/$LABEL.diskstats.before"
T0=$(date +%s.%N); date -u +%FT%TZ > "$OUT/$LABEL.start"
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy - a previous server is still running" | tee "$OUT/$LABEL.rc"; kill $SP; exit 1; fi
~/llama.cpp-qwen38/build/bin/llama-server --port $PORT --host 127.0.0.1 \
    --log-file "$OUT/$LABEL.server.log" "$@" < /dev/null > "$OUT/$LABEL.server.stdout" 2>&1 & SRV=$!
# wait for the model to load
until curl -sf -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do
    if ! kill -0 $SRV 2>/dev/null; then echo "server died" > "$OUT/$LABEL.rc"; kill $SP; exit 1; fi
    sleep 2
done
T1=$(date +%s.%N); echo "$T1 - $T0" | bc > "$OUT/$LABEL.load_seconds"
grep -E " (md0|sda|sdb) " /proc/diskstats > "$OUT/$LABEL.diskstats.loaded"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/$LABEL.vram.loaded"
python3 - "$OUT/$LABEL" "$S/prompt.txt" "$NPRED" "$PORT" <<'PY'
import json, sys, time, urllib.request
out, prompt_path, n, port = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({"prompt": open(prompt_path).read(), "n_predict": n, "temperature": 0,
                   "cache_prompt": False, "stream": False}).encode()
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    doc = json.loads(r.read())
doc["_wall_seconds"] = time.time() - t0
json.dump(doc, open(out + ".completion.json", "w"), indent=1)
t = doc.get("timings", {})
print(f"prompt {t.get('prompt_n')} tok in {t.get('prompt_ms',0)/1000:.1f}s ({t.get('prompt_per_second',0):.3f} t/s); "
      f"gen {t.get('predicted_n')} tok in {t.get('predicted_ms',0)/1000:.1f}s ({t.get('predicted_per_second',0):.3f} t/s)")
PY
T2=$(date +%s.%N); date -u +%FT%TZ > "$OUT/$LABEL.end"
grep -E " (md0|sda|sdb) " /proc/diskstats > "$OUT/$LABEL.diskstats.after"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/$LABEL.vram.after"
grep -E "VmHWM|VmRSS" /proc/$SRV/status > "$OUT/$LABEL.time"; awk '{print "majflt", $12, "minflt", $10}' /proc/$SRV/stat >> "$OUT/$LABEL.time"
kill -TERM $SRV; for i in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -0 $SRV 2>/dev/null && kill -KILL $SRV; wait $SRV 2>/dev/null || true
kill $SP 2>/dev/null; wait $SP 2>/dev/null || true
echo 0 > "$OUT/$LABEL.rc"; echo "$LABEL done"

import pathlib, re
S = "/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b"
p = pathlib.Path(S + "/serve_run.sh"); s = p.read_text()
old = '''/usr/bin/time -v -o "$OUT/$LABEL.time" ~/llama.cpp-qwen38/build/bin/llama-server --port $PORT --host 127.0.0.1 \\
    --log-file "$OUT/$LABEL.server.log" "$@" < /dev/null > "$OUT/$LABEL.server.stdout" 2>&1 & SRV=$!
'''
new = '''if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy - a previous server is still running" | tee "$OUT/$LABEL.rc"; kill $SP; exit 1; fi
~/llama.cpp-qwen38/build/bin/llama-server --port $PORT --host 127.0.0.1 \\
    --log-file "$OUT/$LABEL.server.log" "$@" < /dev/null > "$OUT/$LABEL.server.stdout" 2>&1 & SRV=$!
'''
if old in s:
    s = s.replace(old, new)
old2 = '''kill $SRV; wait $SRV 2>/dev/null || true
'''
new2 = '''grep -E "VmHWM|VmRSS" /proc/$SRV/status > "$OUT/$LABEL.time"; awk '{print "majflt", $12, "minflt", $10}' /proc/$SRV/stat >> "$OUT/$LABEL.time"
kill -TERM $SRV; for i in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -0 $SRV 2>/dev/null && kill -KILL $SRV; wait $SRV 2>/dev/null || true
'''
if old2 in s:
    s = s.replace(old2, new2)
p.write_text(s)
a = pathlib.Path(S + "/analyze_base.py"); t = a.read_text()
t = t.replace('        if "Maximum resident" in line: out["max_rss_gb"] = int(line.split()[-1]) * 1024 / GB',
              '        if line.startswith("VmHWM"): out["max_rss_gb"] = int(line.split()[1]) * 1024 / GB')
t = t.replace('        if "Major (requiring" in line: out["major_faults"] = int(line.split()[-1])',
              '        if line.startswith("majflt"): out["major_faults"] = int(line.split()[1])')
a.write_text(t)
print("runner+analyzer patched")

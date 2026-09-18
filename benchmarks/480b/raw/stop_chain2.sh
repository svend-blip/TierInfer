#!/usr/bin/env bash
# kill chain2 and its children by pid (never by a pattern that could match the caller), then the orphaned server
S=/tmp/claude-1000/-home-svend-DPMtF-WebUI/ae95e412-4516-40b3-8970-4d2d7ebbba2b/scratchpad/480b
me=$$; par=$PPID
for pat in "chain2.sh" "serve_run.sh" "sample.py" "urllib.request"; do
  for pid in $(ps -eo pid,args | awk -v p="$pat" -v me="$me" -v par="$par" '$0 ~ p && $1 != me && $1 != par {print $1}'); do
    echo "kill $pid ($pat)"; kill -TERM "$pid" 2>/dev/null
  done
done
sleep 1
for pid in $(ps -eo pid,args | awk '/llama-server --port 8931/ && !/awk/ {print $1}'); do echo "kill server $pid"; kill -TERM "$pid"; done
for i in $(seq 1 60); do ss -ltn | grep -q ":8931 " || break; sleep 1; done
ss -ltn | grep ":8931 " && echo "PORT STILL BUSY" || echo "port 8931 free"
rm -f $S/base/native-ngl0-cold2.*
ps -eo pid,args | grep -E "llama-server|serve_run|chain2|sample.py" | grep -v grep | cut -c1-100

#!/usr/bin/env python3
"""Summarise serve_run.sh outputs: one row per run, phases split by timestamps.
usage: analyze_base.py DIR [DIR...]  -> prints a markdown table and writes DIR/summary.json"""
import csv, glob, json, os, statistics, sys
GB = 1e9

def diskstats(path):
    d = {}
    for line in open(path):
        f = line.split()
        d[f[2]] = (int(f[3]), int(f[5]) * 512, int(f[6]))   # reads, bytes, ms
    return d

def delta(a, b, dev):
    r = b[dev][0] - a[dev][0]; by = b[dev][1] - a[dev][1]; ms = b[dev][2] - a[dev][2]
    return {"reads": r, "bytes": by, "avg_kb": (by / r / 1024) if r else 0, "await_ms": (ms / r) if r else 0}

def phase(samples, t0, t1, dev):
    rows = [r for r in samples if t0 <= float(r["t"]) <= t1]
    if not rows: return {}
    def m(k): return statistics.mean(float(r[k]) for r in rows)
    return {"n": len(rows), "MBps": m(f"{dev}_MBps"), "rps": m(f"{dev}_rps"), "avg_kb": m(f"{dev}_avg_kb"),
            "await_ms": m(f"{dev}_await_ms"), "util": m(f"{dev}_util"), "cpu": m("cpu_busy_frac"),
            "mem_avail_gb": m("mem_available") / GB, "cached_gb": m("cached") / GB, "vram_gb": m("vram_used") / GB}

def run(stem):
    out = {"label": os.path.basename(stem)}
    try:
        comp = json.load(open(stem + ".completion.json")); t = comp["timings"]
    except FileNotFoundError:
        return None
    out.update(prompt_n=t["prompt_n"], prompt_s=t["prompt_ms"] / 1000, prompt_tps=t["prompt_per_second"],
               gen_n=t["predicted_n"], gen_s=t["predicted_ms"] / 1000, gen_tps=t["predicted_per_second"],
               wall_completion_s=comp["_wall_seconds"], text=comp["content"][:80])
    out["load_s"] = float(open(stem + ".load_seconds").read())
    before, loaded, after = (diskstats(stem + s) for s in (".diskstats.before", ".diskstats.loaded", ".diskstats.after"))
    out["load_io"] = {d: delta(before, loaded, d) for d in ("md0", "sda", "sdb")}
    out["infer_io"] = {d: delta(loaded, after, d) for d in ("md0", "sda", "sdb")}
    for line in open(stem + ".time"):
        if line.startswith("VmHWM"): out["max_rss_gb"] = int(line.split()[1]) * 1024 / GB
        if line.startswith("majflt"): out["major_faults"] = int(line.split()[1])
    samples = list(csv.DictReader(open(stem + ".samples.csv")))
    import datetime
    def ts(p): return datetime.datetime.strptime(open(p).read().strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    t_start, t_end = ts(stem + ".start"), ts(stem + ".end")
    t_loaded = t_start + out["load_s"]
    t_gen0 = t_end - out["gen_s"]          # generation is the tail of the completion
    t_prompt0 = t_gen0 - out["prompt_s"]
    out["phase_load"] = phase(samples, t_start, t_loaded, "md0")
    out["phase_prompt"] = phase(samples, t_prompt0, t_gen0, "md0")
    out["phase_gen"] = phase(samples, t_gen0, t_end, "md0")
    out["phase_gen_sda"] = phase(samples, t_gen0, t_end, "sda")
    g = out["phase_gen"]
    if g:
        out["gen_gb_per_token"] = g["MBps"] * out["gen_s"] / 1000 / out["gen_n"]
        out["gen_reads_per_token"] = g["rps"] * out["gen_s"] / out["gen_n"]
    for f in (".vram.loaded", ".vram.after"):
        if os.path.exists(stem + f): out["vram" + f.split(".")[-1]] = open(stem + f).read().strip()
    return out

rows = []
for d in sys.argv[1:]:
    for c in sorted(glob.glob(os.path.join(d, "*.completion.json"))):
        r = run(c[:-len(".completion.json")])
        if r: rows.append(r)
    json.dump(rows, open(os.path.join(d, "summary.json"), "w"), indent=1)
print("| run | load s | prompt t/s | gen t/s | gen: md0 GB/tok | reads/tok | md0 KB | sda KB | await ms | util | CPU | RSS GB | major faults |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for r in rows:
    g, s = r.get("phase_gen", {}), r.get("phase_gen_sda", {})
    print(f"| {r['label']} | {r['load_s']:.0f} | {r['prompt_tps']:.3f} | {r['gen_tps']:.3f} | "
          f"{r.get('gen_gb_per_token', 0):.2f} | {r.get('gen_reads_per_token', 0):.0f} | {g.get('avg_kb', 0):.0f} | "
          f"{s.get('avg_kb', 0):.0f} | {g.get('await_ms', 0):.2f} | {g.get('util', 0):.2f} | {g.get('cpu', 0):.2f} | "
          f"{r.get('max_rss_gb', 0):.0f} | {r.get('major_faults', 0)} |")

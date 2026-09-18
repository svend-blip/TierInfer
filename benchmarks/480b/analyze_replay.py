#!/usr/bin/env python3
"""One row per replay arm from its summary.json — the §4 table of VALIDATION-480B.md."""
import glob, json, os, sys
GB = 1024 ** 3
paths = sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/replay-out/480b", "*.summary.json")))
only = sys.argv[2:]  # optional label prefixes
print("| arm | tokens | ms/token median (min–max) | wait ms | RAM cache hit | md0 GB/token | md0 reads/token | md0 KB | sda KB | prefetch issued / useful / late / wasted | wasted GB | stalls/token | VRAM hit | verified |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for p in paths:
    d = json.load(open(p)); l = d["label"]
    if only and not any(l.startswith(o) for o in only):
        continue
    pt, pf, c, dev, v = d["per_token"], d["prefetch"], d["cache"], d["device"], d.get("vram")
    n = d["tokens_replayed"] or 1
    mem = dev["members"].get("sda", {})
    print(f"| {l} | {n} | {pt['wall_ms_median']:.0f} ({pt['wall_ms_min']:.0f}–{pt['wall_ms_max']:.0f}) | {pt['wait_ms_median']:.0f} | "
          f"{c['hit_rate']:.1%} | {pt['dev_bytes_median'] / GB:.2f} | {pt['dev_reads_median']:.0f} | {dev['mean_read_bytes'] / 1024:.0f} | "
          f"{mem.get('mean_read_bytes', 0) / 1024:.0f} | {pf['issued']} / {pf['useful']} / {pf['late']} / {pf['cancelled']} | "
          f"{pf['wasted_bytes'] / GB:.1f} | {pf['stalls'] / n:.0f} | "
          f"{(f"{v['hit_rate']:.1%}" if v else '—')} | "
          f"{d['verification']['verified']}/{d['verification']['mismatches']} mism. |")

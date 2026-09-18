"""Combine freetoken_ab.py results into one table: FreeToken's numbers, TierInfer's, the device's.

    python benchmarks/480b/freetoken_table.py benchmarks/freetoken-out flashnext
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GiB = 1024 ** 3


def main(out: Path, prefix: str) -> int:
    runs = []
    for p in sorted(out.glob(f"{prefix}*.json")):
        d = json.loads(p.read_text())
        d["_label"] = p.stem
        tel = out / f"{p.stem}.telemetry.jsonl"
        if tel.exists():
            rows = [json.loads(l) for l in tel.read_text().splitlines() if l.strip()]
            toks = [r for r in rows if r.get("kind") == "token"]
            close = [r for r in rows if r.get("kind") == "run.close"]
            if toks:
                first, gen = toks[0], toks[2:]          # event 1 = prefill; event 2 pairs prefill's scoring with step 1
                hr = [t["hits"] / (t["hits"] + t["misses"]) for t in gen if t["hits"] + t["misses"]]
                d["tier"] = {"prefill_gb": first["bytes_copied"] / 1e9, "prefill_faults": first["faults"],
                             "hit_rate": statistics.median(hr) if hr else None,
                             "mb_per_token": statistics.median(t["bytes_copied"] for t in gen) / 1e6 if gen else None,
                             "faults_per_token": statistics.median(t["faults"] for t in gen) if gen else None,
                             "evictions_per_token": statistics.median(t["evictions"] for t in gen) if gen else None,
                             "wall_ms_per_token": statistics.median(t["wall_ms"] for t in gen) if gen else None,
                             "prefetch": (close[-1].get("loader.prefetch_issued"), close[-1].get("loader.prefetch_useful"),
                                          close[-1].get("loader.prefetch_late"), close[-1].get("loader.prefetch_wasted")) if close else None}
        runs.append(d)
    if not runs:
        return 1
    ref = runs[0]["text"]
    lines = [f"# {prefix}: FreeToken native vs TierInfer-tiered CPU-executor layers", "",
             "| run | arm | tier GB | depth | load s | wall s | FT decode t/s | infer GiB | reads | mean KB | load GiB | output = first |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for r in runs:
        io, th = r["io_infer"], r["stats_after"].get("throughput", {})
        if r.get("arm") != "tiered":
            depth = "—"
        elif r.get("depth") is not None:
            depth = str(r["depth"])
        else:                                   # older records: the label says (…-t8d8-…)
            import re
            m = re.search(r"d(\d+)-tiered", r["_label"])
            depth = m.group(1) if m else "0"
        lines.append(f"| {r['_label']} | {r.get('arm', '?')} | {r.get('tier_gb') or '—'} | {depth} | {r['load_s']:.0f} | {r['wall_s']:.1f} | "
                     f"{th.get('decode_tps', '—')} | {io['bytes'] / GiB:.2f} | {io['reads']} | {io['mean_read_bytes'] / 1024:.0f} | "
                     f"{r['io_load']['bytes'] / GiB:.1f} | {'yes' if r['text'] == ref else 'NO'} |")
    lines += ["", "| run | prefill GB through tier | prefill faults | hit rate (decode) | MB/token | faults/token | evictions/token | ms/token | prefetch issued/useful/late/wasted |",
              "|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    for r in runs:
        t = r.get("tier")
        if not t:
            continue
        lines.append(f"| {r['_label']} | {t['prefill_gb']:.1f} | {t['prefill_faults']} | "
                     f"{'—' if t['hit_rate'] is None else f'{100 * t['hit_rate']:.1f}%'} | {t['mb_per_token']:.0f} | {t['faults_per_token']:.0f} | "
                     f"{t['evictions_per_token']:.0f} | {t['wall_ms_per_token']:.0f} | {'/'.join(str(x) for x in t['prefetch']) if t['prefetch'] else '—'} |")
    lines += ["", f"Greedy output identical across all runs and arms: **{len({r['text'] for r in runs}) == 1}**", ""]
    (out / f"{prefix}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2]))

"""Combine loader_ab.py results (one JSON per run) into one A/B table.

    python benchmarks/480b/harness_table.py benchmarks/loader-out q480b-ncmoe60
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GB = 1024 ** 3   # GiB, as the harness prints them


def main(out: Path, stem: str) -> int:
    runs = []
    for p in sorted(out.glob(f"{stem}-*.json")):
        if "crash" in p.name:
            continue
        d = json.loads(p.read_text())
        arm, rep = p.stem[len(stem) + 1:].rsplit("-", 1)
        d["arm"], d["rep"] = arm, int(rep)
        tel = out / f"{p.stem}.telemetry.jsonl"
        if tel.exists():
            rows = [json.loads(l) for l in tel.read_text().splitlines() if l.strip()]
            close = [r for r in rows if r.get("kind") == "run.close"]
            toks = [r for r in rows if r.get("kind") == "token"][2:]      # after the prompt batch
            if close:
                c = close[-1]
                d["loader"] = {k.replace("loader.", ""): v for k, v in c.items() if k.startswith("loader.")}
            if toks:
                hr = [t["hits"] / (t["hits"] + t["misses"]) for t in toks if t["hits"] + t["misses"]]
                d["per_token"] = {"hit_rate": statistics.median(hr) if hr else None,
                                  "faults": statistics.median(t["faults"] for t in toks),
                                  "bytes": statistics.median(t["bytes_copied"] for t in toks),
                                  "evictions": statistics.median(t["evictions"] for t in toks),
                                  "wall_ms": statistics.median(t["wall_ms"] for t in toks)}
        runs.append(d)
    if not runs:
        print("no runs", file=sys.stderr)
        return 1
    ref = next((r for r in runs if r["arm"] == "native" and r["rep"] == 1), runs[0])
    lines = [f"# {stem}: native vs loader ({' '.join(ref['cmd'][ref['cmd'].index('-c') + 2:])})", "",
             "| arm | run | load s | prompt t/s | gen t/s | infer GiB | infer reads | mean KB | await ms | tokens = native #1 |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for r in sorted(runs, key=lambda r: (r["rep"], r["arm"] != "native")):
        io = r["io_infer"]
        lines.append(f"| {r['arm']} | {r['rep']} | {r['load_s']:.0f} | {r['prompt_tps']:.3f} | {r['gen_tps']:.3f} | "
                     f"{io['bytes'] / GB:.1f} | {io['reads']} | {io['mean_read_bytes'] / 1024:.0f} | "
                     f"{io['await_ms']:.1f} | {'yes' if r['content'] == ref['content'] else 'NO'} |")
    for arm in ("native", "loader"):
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        g = [r["gen_tps"] for r in rs]
        lines.append(f"| **{arm} median** | {len(rs)} | {statistics.median(r['load_s'] for r in rs):.0f} | "
                     f"{statistics.median(r['prompt_tps'] for r in rs):.3f} | **{statistics.median(g):.3f}** ({min(g):.3f}–{max(g):.3f}) | "
                     f"{statistics.median(r['io_infer']['bytes'] for r in rs) / GB:.1f} | "
                     f"{statistics.median(r['io_infer']['reads'] for r in rs):.0f} | | | |")
    lt = [r for r in runs if r["arm"] == "loader" and "loader" in r]
    if lt:
        lines += ["", "| loader run | hit rate (gen, median) | faults/token | MB copied/token | evictions/token | wall ms/token | run: faults | resident-page faults | repaired | evictions | UNMAPs | forgotten |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in lt:
            L, pt = r["loader"], r.get("per_token", {})
            lines.append(f"| {r['rep']} | {100 * (pt.get('hit_rate') or 0):.1f}% | {pt.get('faults', 0):.0f} | "
                         f"{pt.get('bytes', 0) / 1e6:.0f} | {pt.get('evictions', 0):.0f} | {pt.get('wall_ms', 0):.0f} | "
                         f"{L.get('faults')} | {L.get('faults_resident')} | {L.get('faults_repaired')} | {L.get('evictions')} | "
                         f"{L.get('unmaps')} | {L.get('forgotten')} |")
    same = len({r["content"] for r in runs}) == 1
    lines += ["", f"Greedy tokens identical across all runs and arms: **{same}**", ""]
    (out / f"{stem}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2]))

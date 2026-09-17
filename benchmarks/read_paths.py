#!/usr/bin/env python3
"""Compare ways of getting the same expert off NVMe.

Answers one question with measurement rather than argument: how much of the
cost in the baseline experiment was the bytes, and how much was asking for
them 4 KB at a time?

    python benchmarks/read_paths.py <model.gguf> [--layer N] [--expert N]

Each mode reads the identical byte ranges. Cold modes evict those ranges from
the page cache first, so the drive is actually touched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.index import load  # noqa: E402
from tierinfer.storage import StorageBackend, coalesce  # noqa: E402

MB = 1024 ** 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--expert", type=int, default=37)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    ix = load(a.model)
    ref = ix.expert(a.layer, a.expert)
    ranges = list(ref.ranges)
    merged = coalesce(ranges)

    results: dict[str, dict] = {}
    with StorageBackend(a.model) as s:
        for label, fn, cold in (
            ("cold_object_reads", lambda: s.read(ranges)[1], True),
            ("cold_coalesced", lambda: s.read(merged)[1], True),
            ("cold_4k_pages", lambda: s.read_paged(ranges, 4096)[1], True),
            ("warm_object_reads", lambda: s.read(ranges)[1], False),
        ):
            samples = []
            for _ in range(a.repeat):
                if cold:
                    s.evict(ranges)
                stat = fn()
                samples.append(stat)
            results[label] = {
                "median_ms": statistics.median(x.seconds for x in samples) * 1000,
                "gigabytes_per_second": statistics.median(x.bytes_per_second for x in samples) / 1e9,
                "operations": samples[0].operations,
                "mean_operation_bytes": samples[0].mean_operation_bytes,
                "bytes": samples[0].nbytes,
            }

    report = {
        "model": str(ix.gguf.path),
        "layer": a.layer,
        "expert": a.expert,
        "expert_megabytes": ref.nbytes / MB,
        "ranges": len(ranges),
        "coalesced_ranges": len(merged),
        "modes": results,
    }
    base = results["cold_object_reads"]
    pages = results["cold_4k_pages"]
    report["paging_penalty"] = {
        "times_slower": pages["median_ms"] / base["median_ms"],
        "times_more_operations": pages["operations"] / base["operations"],
    }

    if a.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{report['expert_megabytes']:.2f} MB — layer {a.layer} expert {a.expert}, "
          f"{len(ranges)} ranges, median of {a.repeat}\n")
    print(f"{'mode':<20} {'ms':>8} {'GB/s':>7} {'ops':>7} {'mean op':>10}")
    for name, r in results.items():
        print(f"{name:<20} {r['median_ms']:8.2f} {r['gigabytes_per_second']:7.2f} "
              f"{r['operations']:7d} {r['mean_operation_bytes'] / 1024:9.1f}K")
    pp = report["paging_penalty"]
    print(f"\n4 KB paging costs {pp['times_slower']:.1f}x the time and "
          f"{pp['times_more_operations']:.0f}x the operations for the same bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

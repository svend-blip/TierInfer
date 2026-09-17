#!/usr/bin/env python3
"""Does advising from inside the loop beat advising from beside it?

The horizon measurement says a residency decision has to be made on a one- to
two-token horizon, because by the eighth token a third of the model has been
touched. A helper process cannot act that fast. `cb_eval` can: it fires once
per MoE layer during the forward pass, so when layer L's routing is known,
layers L+1 and L+2 have not run yet.

Both arms run the same binary on the same prompt under the same ceiling and
both write a trace, so the only difference is whether the callback also
advises the page cache. The guess is the plainest one the data supports —
what those layers routed to for the previous token, which shares 38% of its
experts with this one.

    python benchmarks/inloop.py MODEL.gguf MAP --limit 32 --tokens 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from tierinfer.bench import measure, memory_controller_available, write_report  # noqa: E402

GB = 1024 ** 3
ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "tierinfer-trace"

PROMPT = ("Write a detailed technical explanation of how mixture-of-experts "
          "routing works in modern transformer language models.")


def argv_for(model: Path, tokens: int, threads: int, out: Path,
             expert_map: Path | None, horizon: int, evict_after: int) -> list[str]:
    argv = [str(TOOL), "-m", str(model), "-ngl", "0", "-t", str(threads),
            "-n", str(tokens), "-c", "4096", "-p", PROMPT, "-o", str(out)]
    if expert_map is not None and (horizon > 0 or evict_after > 0):
        argv += ["--expert-map", str(expert_map)]
        if horizon > 0:
            argv += ["--horizon", str(horizon)]
        if evict_after > 0:
            argv += ["--evict-after", str(evict_after)]
    return argv


def row(label: str, m, tokens: int) -> str:
    per_token = m.execution.wall_seconds / tokens if tokens else 0.0
    return (f"{label:<20}{m.execution.wall_seconds:>9.0f}s{per_token:>11.2f}"
            f"{m.disk.gb_read:>11.1f}{m.disk.bandwidth_gbps:>9.2f}"
            f"{m.disk.iops:>10,.0f}{m.disk.mean_read_bytes / 1024:>9.0f}K"
            f"{m.residency_after.fraction * 100:>10.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("expert_map", type=Path)
    ap.add_argument("--limit", type=float, default=32.0)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--arm", action="append", default=[],
                    help="an arm as horizon:evict_after, e.g. 0:2 or 2:2; repeatable")
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "inloop-report.json")
    a = ap.parse_args()
    arms: list[tuple[int, int]] = []
    for spec in (a.arm or ["2:0"]):
        parts = spec.split(":")
        h = int(parts[0] or 0)
        ev = int(parts[1] or 0) if len(parts) > 1 else 0
        unused = len(parts) > 2 and parts[2] in ("u", "unused", "1")
        arms.append((h, ev, unused))

    if not TOOL.exists():
        print(f"no {TOOL}; run tools/trace/build.sh first", file=sys.stderr)
        return 2
    if not memory_controller_available():
        print("the memory controller is not delegated here", file=sys.stderr)
        return 2

    scratch = ROOT / "build"
    print(f"model {a.model.name}, ceiling {a.limit:g} GB, {a.tokens} tokens, "
          f"both arms writing a trace\n")
    print(f"{'arm':<20}{'wall':>10}{'s/token':>11}{'GB read':>11}{'GB/s':>9}"
          f"{'IOPS':>10}{'mean rd':>10}{'resident':>11}")

    results, plain = [], None
    for horizon, evict, unused in [(0, 0, False)] + arms:
        tag = f"{horizon}-{evict}{'u' if unused else ''}"
        label = "no assist" if (horizon, evict) == (0, 0) else \
                f"fetch {horizon} / free {evict}{' +unused' if unused else ''}"
        argv = argv_for(a.model, a.tokens, a.threads,
                        scratch / f"inloop-{tag}.jsonl",
                        a.expert_map if (horizon or evict) else None, horizon, evict)
        if unused:
            argv.append("--release-unused")
        m = measure(f"inloop-{tag}", a.model, argv, cold=True,
                    memory_max_bytes=int(a.limit * GB), timeout=a.budget)
        results.append(m)
        print(row(label, m, a.tokens), flush=True)
        if (horizon, evict) == (0, 0):
            plain = m
        elif plain is not None and not m.execution.timed_out and not plain.execution.timed_out:
            faster = plain.execution.wall_seconds / max(m.execution.wall_seconds, 1e-9)
            print(f"{'':<20}{faster:>9.2f}x of no-assist wall, "
                  f"{m.disk.gb_read - plain.disk.gb_read:+.1f} GB read", flush=True)

    print()
    for m in results:
        for note in m.notes:
            print(f"note [{m.condition}]: {note}")
        for line in m.execution.stdout.splitlines():
            if "advised" in line or "never being routed" in line:
                print(f"[{m.condition}] {line.strip()}")
    write_report(results, a.out)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

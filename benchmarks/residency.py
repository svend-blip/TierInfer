#!/usr/bin/env python3
"""Does keeping the floor warm help a run that does not fit?

The baseline measured llama.cpp under a 32 GB ceiling on a 56.47 GB model:
0.80 tokens/s, and 113.6 GB read for a 56.5 GB file, because pages are
evicted before they are used again. 4.55 GB of that file is touched by every
single token — attention, the shared expert, the router, the embeddings —
and under pressure the kernel reclaims it by age like anything else, with
51.91 GB of routed experts doing the ageing.

So this runs the same condition twice, identically, except that the second
has a helper touching the floor every couple of seconds. Both the helper and
llama.cpp run inside one cgroup scope, so the helper's page faults are
charged against the same ceiling; a helper outside it would quietly widen the
limit it is supposed to work under.

    python benchmarks/residency.py MODEL.gguf --limit 32 --tokens 8

If the assisted run is not faster, external residency shaping of this kind
does not work on this model, and that is the result.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from tierinfer.bench import measure, memory_controller_available, write_report  # noqa: E402

GB = 1024 ** 3
ROOT = Path(__file__).resolve().parent.parent

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("baseline", ROOT / "benchmarks" / "baseline.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def assisted_argv(llama_argv: list[str], model: Path, interval: float) -> list[str]:
    """llama.cpp and the floor helper in one shell, so one scope holds both."""
    pin = f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'tools' / 'pin_floor.py'))} " \
          f"{shlex.quote(str(model))} --interval {interval}"
    run = " ".join(shlex.quote(a) for a in llama_argv)
    return ["bash", "-c", f"{pin} & helper=$!; {run}; rc=$?; kill $helper 2>/dev/null; exit $rc"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--llama", type=Path, default=baseline.DEFAULT_LLAMA)
    ap.add_argument("--limit", type=float, action="append", default=[])
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=1200.0)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "residency-report.json")
    a = ap.parse_args()
    limits = a.limit or [32.0]

    if not memory_controller_available():
        print("the memory controller is not delegated to this user; this "
              "experiment cannot be run here", file=sys.stderr)
        return 2

    prompt = "Explain what a mixture-of-experts layer does."
    llama_argv = baseline.argv_for(a.llama, a.model, a.tokens, a.threads, prompt)

    print(f"model {a.model.name}  {a.model.stat().st_size / GB:.2f} GB")
    print(f"helper refreshes the floor every {a.interval:g}s, inside the same scope\n")
    print(baseline.HEADER)

    results = []
    for gb in limits:
        for label, argv in (("plain", llama_argv),
                            ("floor-pinned", assisted_argv(llama_argv, a.model, a.interval))):
            m = measure(f"{label}@{gb:.0f}GB", a.model, argv, cold=True,
                        memory_max_bytes=int(gb * GB), timeout=a.budget)
            results.append(m)
            print(baseline.row(m), flush=True)

    print()
    for m in results:
        for note in m.notes:
            print(f"note [{m.condition}]: {note}")

    for gb in limits:
        plain = next((m for m in results if m.condition == f"plain@{gb:.0f}GB"), None)
        pinned = next((m for m in results if m.condition == f"floor-pinned@{gb:.0f}GB"), None)
        if not (plain and pinned):
            continue
        pt, at = baseline.throughput(plain.execution.stdout), baseline.throughput(pinned.execution.stdout)
        print(f"\nat {gb:.0f} GB:")
        if pt and at:
            # A ratio formatted as a percentage reads as a change: 1.12
            # printed as "+112%" says "more than doubled" for a 12% gain.
            print(f"  throughput {pt:.2f} -> {at:.2f} t/s  "
                  f"({at / pt - 1:+.1%}, {at / pt:.2f}x)")
        else:
            print(f"  throughput: plain {pt}, pinned {at} — one of them produced none")
        print(f"  read       {plain.disk.gb_read:.1f} -> {pinned.disk.gb_read:.1f} GB")
        print(f"  mean read  {plain.disk.mean_read_bytes / 1024:.0f}K -> "
              f"{pinned.disk.mean_read_bytes / 1024:.0f}K")

    write_report(results, a.out)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

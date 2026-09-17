#!/usr/bin/env python3
"""The baseline experiment, automated: what does a model cost when it does
not fit in the memory you give it?

Four conditions, each measured the same way:

    warm          the page cache holds the model; this is the RAM ceiling
    cold          the cache is dropped first; this is the first-run cost
    constrained   cold, plus a memory ceiling below the model's size

Each condition reports throughput, the NVMe traffic behind it, how much of
the model ended up resident, and — where a ceiling applies — how much memory
the run actually took.

    python benchmarks/baseline.py MODEL.gguf [--limit 32] [--limit 16] \\
        [--tokens 64] [--budget 3600] [--out report.json]

The conditions run cheapest-to-truth first: warm establishes what the machine
can do at all, so a cold or constrained number has something to be a fraction
of.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tierinfer.bench import (  # noqa: E402
    BenchError, Measurement, drop_cache, measure, memory_controller_available,
    residency, warm_cache, write_report,
)

GB = 1024 ** 3
DEFAULT_LLAMA = Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli"

# llama.cpp prints "eval time = 1234.56 ms / 64 runs ( 19.29 ms per token, 51.84 tokens per second)"
EVAL_RE = re.compile(r"eval time =.*?([\d.]+)\s+tokens per second", re.S)
LOAD_RE = re.compile(r"load time =\s*([\d.]+)\s*ms")


def throughput(stdout: str) -> float | None:
    """Generation tokens per second as llama.cpp measured it, if it got that far."""
    hits = EVAL_RE.findall(stdout)
    return float(hits[-1]) if hits else None


def load_seconds(stdout: str) -> float | None:
    m = LOAD_RE.search(stdout)
    return float(m.group(1)) / 1000.0 if m else None


def argv_for(llama: Path, model: Path, tokens: int, threads: int, prompt: str) -> list[str]:
    return [str(llama), "-m", str(model), "-ngl", "0", "-t", str(threads),
            "-n", str(tokens), "-p", prompt, "--no-warmup", "-no-cnv"]


def row(m: Measurement) -> str:
    tps = throughput(m.execution.stdout)
    load = load_seconds(m.execution.stdout)
    peak = m.execution.peak_memory_bytes
    return (f"{m.condition:<16}"
            f"{(f'{tps:.2f}' if tps else '—'):>10}"
            f"{(f'{load:.0f}s' if load else '—'):>9}"
            f"{m.execution.wall_seconds:>9.0f}s"
            f"{m.disk.gb_read:>10.1f}"
            f"{m.disk.bandwidth_gbps:>9.2f}"
            f"{m.disk.iops:>10,.0f}"
            f"{m.disk.mean_read_bytes / 1024:>9.0f}K"
            f"{m.disk.await_ms:>9.2f}"
            f"{m.residency_after.fraction * 100:>9.0f}%"
            f"{(f'{peak / GB:.1f}' if peak else '—'):>9}")


HEADER = (f"{'condition':<16}{'tok/s':>10}{'load':>9}{'wall':>10}{'GB read':>10}"
          f"{'GB/s':>9}{'IOPS':>10}{'mean rd':>10}{'await':>9}{'resident':>10}{'peak GB':>9}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--llama", type=Path, default=DEFAULT_LLAMA)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--prompt", default="Explain what a mixture-of-experts layer does.")
    ap.add_argument("--limit", type=float, action="append", default=[],
                    help="a memory ceiling in GB; repeatable")
    ap.add_argument("--budget", type=float, default=3600.0,
                    help="seconds any single condition may take before it is cut")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("baseline-report.json"))
    a = ap.parse_args()

    if not a.model.exists():
        print(f"no such model: {a.model}", file=sys.stderr)
        return 2
    if not a.llama.exists():
        print(f"no llama.cpp binary at {a.llama}", file=sys.stderr)
        return 2

    size = a.model.stat().st_size
    argv = argv_for(a.llama, a.model, a.tokens, a.threads, a.prompt)
    print(f"model {a.model.name}  {size / GB:.2f} GB")
    print(f"command {' '.join(argv)}")
    print(f"budget {a.budget:.0f}s per condition\n")
    print(HEADER)

    results: list[Measurement] = []

    if not a.skip_warm:
        warm_cache(a.model)
        m = measure("warm", a.model, argv, timeout=a.budget)
        results.append(m)
        print(row(m), flush=True)

    m = measure("cold", a.model, argv, cold=True, timeout=a.budget)
    results.append(m)
    print(row(m), flush=True)

    for gb in a.limit:
        if not memory_controller_available():
            print(f"skipping the {gb:.0f} GB ceiling: the memory controller is not "
                  "delegated to this user", file=sys.stderr)
            break
        try:
            m = measure(f"cold+{gb:.0f}GB", a.model, argv, cold=True,
                        memory_max_bytes=int(gb * GB), timeout=a.budget)
        except BenchError as e:
            print(f"skipping the {gb:.0f} GB ceiling: {e}", file=sys.stderr)
            continue
        results.append(m)
        print(row(m), flush=True)

    print()
    for m in results:
        for note in m.notes:
            print(f"note [{m.condition}]: {note}")
    warm = next((m for m in results if m.condition == "warm"), None)
    if warm and throughput(warm.execution.stdout):
        base = throughput(warm.execution.stdout)
        for m in results:
            t = throughput(m.execution.stdout)
            if t and m is not warm:
                print(f"{m.condition} runs at {t / base:.2f}x of warm")

    write_report(results, a.out)
    print(f"\nwrote {a.out}")
    print("mean read size is a mean, not a distribution: a histogram needs "
          "blktrace, which needs root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

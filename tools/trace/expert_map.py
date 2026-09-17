#!/usr/bin/env python3
"""Emit the expert byte map the in-loop assistant reads.

GGUF parsing stays in Python, where it is tested and where a wrong offset
fails a test rather than a run. The C++ side gets a flat table it cannot
misread:

    # tierinfer-expert-map 1
    # layers <n> experts <n>
    <layer> <expert> <file_offset> <nbytes>

One line per expert per layer, ascending. The assistant loads it once and
never parses a model file itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tierinfer.gguf import GGUFError, read_gguf  # noqa: E402
from tierinfer.index import ModelIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()

    ix = ModelIndex(read_gguf(a.model))
    out = open(a.out, "w") if a.out else sys.stdout
    written = skipped = 0
    try:
        print("# tierinfer-expert-map 1", file=out)
        print(f"# model {a.model.name}", file=out)
        print(f"# layers {len(ix.moe_layers)} experts {ix.expert_count}", file=out)
        for layer in ix.moe_layers:
            for e in range(ix.expert_count):
                try:
                    ref = ix.expert(layer, e)
                except GGUFError:
                    skipped += 1
                    continue
                for r in ref.ranges:
                    print(f"{layer} {e} {r.file_offset} {r.nbytes}", file=out)
                    written += 1
    finally:
        if a.out:
            out.close()
    print(f"expert_map: {written} ranges"
          + (f", {skipped} experts skipped" if skipped else ""), file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

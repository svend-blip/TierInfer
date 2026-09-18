"""``tierinfer inspect <model.gguf>`` — what is in the model and what a token needs."""

from __future__ import annotations

import argparse
import json
import sys

from .gguf import GGUFError
from .index import ModelIndex, load

GB = 1024 ** 3
MB = 1024 ** 2


def _report(ix: ModelIndex) -> dict:
    resident = ix.always_resident_nbytes()
    routed = ix.routed_nbytes()
    per_expert = ix.expert_nbytes()
    return {
        "model": str(ix.gguf.path),
        "shards": len(ix.gguf.files),
        "bytes_on_disk": ix.gguf.nbytes_on_disk,
        "architecture": ix.gguf.architecture,
        "gguf_version": ix.gguf.version,
        "tensors": len(ix.gguf.tensors),
        "layers": ix.block_count,
        "moe_layers": len(ix.moe_layers),
        "experts_per_layer": ix.expert_count,
        "experts_used_per_token": ix.expert_used_count,
        "bytes": {
            "total": resident + routed,
            "always_resident": resident,
            "routed_experts": routed,
            "per_expert": per_expert,
            "working_set_per_token": ix.working_set_nbytes(),
        },
    }


def _print_text(r: dict) -> None:
    b = r["bytes"]
    print(f"model            {r['model']}")
    print(f"architecture     {r['architecture']} (GGUF v{r['gguf_version']}, {r['tensors']} tensors, "
          f"{r['shards']} file{'s' if r['shards'] != 1 else ''})")
    print(f"layers           {r['layers']}, of which {r['moe_layers']} are MoE")
    if r["experts_per_layer"]:
        print(f"experts          {r['experts_per_layer']} per layer, "
              f"{r['experts_used_per_token']} used per token, "
              f"{b['per_expert'] / MB:.2f} MB each")
    print()
    print(f"total            {b['total'] / GB:8.2f} GB")
    print(f"always resident  {b['always_resident'] / GB:8.2f} GB   attention, norms, router, shared experts")
    print(f"routed experts   {b['routed_experts'] / GB:8.2f} GB")
    print(f"working set      {b['working_set_per_token'] / GB:8.2f} GB   what one token actually needs")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tierinfer", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ins = sub.add_parser("inspect", help="report a model's layout and its per-token working set")
    ins.add_argument("model")
    ins.add_argument("--json", action="store_true")

    exp = sub.add_parser("expert", help="print the byte ranges of one routed expert")
    exp.add_argument("model")
    exp.add_argument("layer", type=int)
    exp.add_argument("expert", type=int)
    exp.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    try:
        ix = load(args.model)
        if args.command == "inspect":
            r = _report(ix)
            print(json.dumps(r, indent=2) if args.json else "", end="")
            if not args.json:
                _print_text(r)
            return 0
        ref = ix.expert(args.layer, args.expert)
        if args.json:
            print(json.dumps({
                "layer": ref.layer, "expert": ref.expert, "nbytes": ref.nbytes,
                "ranges": [{"name": r.name, "file_offset": r.file_offset, "nbytes": r.nbytes}
                           for r in ref.ranges],
            }, indent=2))
        else:
            print(f"layer {ref.layer} expert {ref.expert}: {ref.nbytes / MB:.2f} MB "
                  f"in {len(ref.ranges)} ranges")
            for r in ref.ranges:
                print(f"  {r.name:<44} offset {r.file_offset:>14}  +{r.nbytes:>9}")
        return 0
    except (GGUFError, KeyError, OSError) as exc:
        print(f"tierinfer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

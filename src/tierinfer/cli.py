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
    by_layer = ix.expert_nbytes_by_layer()
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
            "per_expert_min": min(by_layer.values()) if by_layer else 0,
            "per_expert_max": max(by_layer.values()) if by_layer else 0,
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
        size = (f"{b['per_expert'] / MB:.2f} MB each" if b["per_expert_min"] == b["per_expert_max"]
                else f"{b['per_expert_min'] / MB:.2f}-{b['per_expert_max'] / MB:.2f} MB each "
                     f"(varies by layer; layer {r['layers'] and 0} has {b['per_expert'] / MB:.2f})")
        print(f"experts          {r['experts_per_layer']} per layer, "
              f"{r['experts_used_per_token']} used per token, {size}")
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

    srv = sub.add_parser("serve", help="answer a preloaded llama.cpp's page faults with whole experts")
    srv.add_argument("model")
    srv.add_argument("--sock", required=True, help="unix socket path; the client gets TIERINFER_SOCK=<this>")
    srv.add_argument("--ram-gb", type=float, default=0.0, help="RAM tier in GiB (0 = autoconfig's share)")
    srv.add_argument("--workers", type=int, default=8)
    srv.add_argument("--depth", type=int, default=0, help="prefetch depth per layer (0 = off)")
    srv.add_argument("--telemetry", default=None, help="JSONL path")
    srv.add_argument("--keep-page-cache", action="store_true",
                     help="do not drop the page cache behind reads (double caching; for diagnosis)")
    srv.add_argument("--quiet", action="store_true")
    srv.add_argument("--capability", default=None,
                     help="a tierinfer.residency capability document (JSON): the RAM tier, prefetch "
                          "depth and workers come from resolving it against this host")

    res = sub.add_parser("resolve", help="resolve a capability document against this host and model, as JSON")
    res.add_argument("capability")
    res.add_argument("model")

    args = p.parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "resolve":
        return _resolve(args)
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


def _serve(args) -> int:
    from .autoconfig import configure
    from .loader import LoaderServer
    from .telemetry import Telemetry
    from pathlib import Path as _P
    ftw = None
    try:
        if _P(args.model).is_dir() and (_P(args.model) / "freetoken_weight.json").exists():
            from .ftw import FTWIndex
            ftw = FTWIndex(args.model)
            ix = None
        else:
            ix = load(args.model)
    except (GGUFError, OSError, ValueError) as exc:
        print(f"tierinfer: {exc}", file=sys.stderr)
        return 1
    depth, workers = args.depth, args.workers
    if ftw is not None:
        # FreeToken's checkpoint: the runtime speaks the protocol itself (tierinfer.client)
        if args.capability:
            print("tierinfer: --capability is resolved against a GGUF index; not for an FTW checkpoint", file=sys.stderr)
            return 1
        ram = int(args.ram_gb * GB)
        if ram <= 0:
            print("tierinfer: give --ram-gb for an FTW checkpoint (no autoconfig for FreeToken banks yet)", file=sys.stderr)
            return 1
        print("tierinfer: serving FreeToken banks; start FreeToken with\n"
              f"  TIERINFER_SOCK={args.sock} (and --moe-cpu-layers for the tiered layers)", file=sys.stderr, flush=True)
        tel = Telemetry(args.telemetry) if args.telemetry else None
        server = LoaderServer(ftw, ram_bytes=ram, workers=workers, depth=depth, telemetry=tel,
                              verbose=not args.quiet, drop_page_cache=not args.keep_page_cache)
        try:
            server.serve(args.sock)
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
        return 0
    if args.capability:
        from .adapters.flowrunner import Capability, CapabilityError, resolve
        try:
            r = resolve(Capability.load(args.capability), ix)
        except CapabilityError as exc:
            print(f"tierinfer: {exc}", file=sys.stderr)
            return 1
        if not r.available:
            print("tierinfer: the capability cannot be provided here:\n  - " + "\n  - ".join(r.refusals),
                  file=sys.stderr)
            return 3
        cfg = r.configuration
        ram = int(args.ram_gb * GB) if args.ram_gb > 0 else cfg.ram_bytes
        depth = args.depth if args.depth else cfg.prefetch_depth
        workers = cfg.stream_workers
        print(r.explain(), file=sys.stderr)
    else:
        ram = int(args.ram_gb * GB) if args.ram_gb > 0 else configure(ix).ram_bytes
    files = ":".join(str(f) for f in ix.gguf.files)
    print("tierinfer: start the runtime with\n"
          f"  LD_PRELOAD=<TierInfer>/build/libtierinfer_mmap.so TIERINFER_SOCK={args.sock} "
          f"TIERINFER_FILES={files}", file=sys.stderr, flush=True)
    tel = Telemetry(args.telemetry) if args.telemetry else None
    server = LoaderServer(ix, ram_bytes=ram, workers=workers, depth=depth, telemetry=tel,
                          verbose=not args.quiet, drop_page_cache=not args.keep_page_cache)
    try:
        server.serve(args.sock)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


def _resolve(args) -> int:
    from .adapters.flowrunner import Capability, CapabilityError, resolve, telemetry_values
    try:
        ix = load(args.model)
        r = resolve(Capability.load(args.capability), ix)
    except (GGUFError, OSError, CapabilityError) as exc:
        print(json.dumps({"available": False, "error": str(exc)}))
        return 1
    out = {"available": r.available, "refusals": r.refusals,
           "configuration": r.configuration.to_dict() if r.configuration else None,
           "telemetry": telemetry_values(r.configuration) if r.configuration else None,
           "files": [str(f) for f in ix.gguf.files]}
    print(json.dumps(out, indent=2))
    return 0 if r.available else 3


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Native llama.cpp against llama.cpp + the TierInfer loader, on the same prompt.

Both arms run the unmodified `llama-server`; the loader arm preloads
`build/libtierinfer_mmap.so` and talks to `tierinfer serve`. Each run
records what the addendum and the acceptance matrix ask for: load time,
prompt and generation tokens/s, time to first token, the generated text
(greedy — the loader arm must produce the *same* tokens as native), device
counters around the completion (bytes, reads, mean request size, await), the
server's RSS and major faults, and for the loader arm its telemetry
(hits, faults, bytes copied, evictions, prefetch useful/late/wasted).

    python benchmarks/loader_ab.py MODEL.gguf --device nvme0n1p2 --repeat 3 \\
        --native-memory-max 32G --loader-ram-gb 27 --extra -ngl 0

The native arm's memory ceiling is a cgroup (`systemd-run --user --scope`),
which is how `BASELINE.md` measured native under pressure; the loader arm
needs none — its RAM tier is the budget. Both arms start cold (the model's
page cache dropped, and the loader drops it behind every read).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from tierinfer.bench import device_for, disk_counters, drop_cache, member_devices  # noqa: E402
from tierinfer.index import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GB = 1024 ** 3
LLAMA = Path(os.environ.get("LLAMA_SERVER", str(Path.home() / "llama.cpp-qwen38/build/bin/llama-server")))
SHIM = ROOT / "build" / "libtierinfer_mmap.so"


def _wait_health(port: int, proc: subprocess.Popen, timeout: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode} before it was healthy")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if b'"ok"' in r.read():
                    return time.time() - t0
        except Exception:  # noqa: BLE001 — not up yet
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def _complete(port: int, prompt: str, n: int) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        doc = json.loads(r.read())
    doc["_wall_seconds"] = time.time() - t0
    return doc


def _proc_status(pid: int) -> dict:
    out = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(("VmHWM", "VmRSS")):
                k, v = line.split(":")
                out[k] = int(v.split()[0]) * 1024
        f = open(f"/proc/{pid}/stat").read().split()
        out["majflt"] = int(f[11])
    except OSError:
        pass
    return out


def _stop(proc: subprocess.Popen, grace: float = 60) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_arm(label: str, model: Path, prompt: str, n_predict: int, extra: list[str], *, port: int,
            device: str, members: list[str], out: Path, loader: dict | None,
            memory_max: str | None, server_log: Path) -> dict:
    env = dict(os.environ)
    prefix: list[str] = []
    if memory_max:
        prefix = ["systemd-run", "--user", "--scope", "--quiet", f"-p", f"MemoryMax={memory_max}",
                  "-p", "MemorySwapMax=0"]
    if loader:
        env.update(LD_PRELOAD=str(SHIM), TIERINFER_SOCK=loader["sock"], TIERINFER_FILES=loader["files"])
    cmd = prefix + [str(LLAMA), "--port", str(port), "--host", "127.0.0.1", "-m", str(model),
                    "-c", "4096", "-t", "32", "--no-warmup", *extra]
    d0 = disk_counters(device)
    m0 = {m: disk_counters(m) for m in members}
    t0 = time.time()
    with open(server_log, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    try:
        load_s = _wait_health(port, proc, timeout=1800)
        d_loaded = disk_counters(device)
        doc = _complete(port, prompt, n_predict)
        d1 = disk_counters(device)
        m1 = {m: disk_counters(m) for m in members}
        # the real server pid: under systemd-run --scope the child is the server itself
        status = _proc_status(proc.pid)
        t = doc["timings"]
        gen_delta = d1 - d_loaded
        return {
            "label": label, "cmd": cmd, "load_s": load_s,
            "prompt_n": t["prompt_n"], "prompt_s": t["prompt_ms"] / 1000, "prompt_tps": t["prompt_per_second"],
            "gen_n": t["predicted_n"], "gen_s": t["predicted_ms"] / 1000, "gen_tps": t["predicted_per_second"],
            "ttft_s": t["prompt_ms"] / 1000, "wall_s": doc["_wall_seconds"], "content": doc["content"],
            "io_load": {"reads": (d_loaded - d0).reads, "bytes": (d_loaded - d0).bytes_read},
            "io_infer": {"reads": gen_delta.reads, "bytes": gen_delta.bytes_read,
                         "mean_read_bytes": gen_delta.mean_read_bytes, "await_ms": gen_delta.await_ms,
                         "members": {m: {"reads": (m1[m] - m0[m]).reads, "bytes": (m1[m] - m0[m]).bytes_read,
                                         "mean_read_bytes": (m1[m] - m0[m]).mean_read_bytes} for m in members}},
            "server": status,
        }
    finally:
        _stop(proc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--prompt", type=Path, default=ROOT / "benchmarks" / "480b" / "raw" / "prompt.txt")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=["-ngl", "0"],
                    help="llama-server arguments for both arms (after --extra)")
    ap.add_argument("--native-memory-max", default=None, help="cgroup ceiling for the native arm, e.g. 32G")
    ap.add_argument("--loader-ram-gb", type=float, default=0.0, help="RAM tier for the loader arm (0 = autoconfig)")
    ap.add_argument("--loader-depth", type=int, default=0)
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--arms", default="native,loader")
    ap.add_argument("--first-rep", type=int, default=1, help="number the runs from here (resuming a series)")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "loader-out")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    ix = load(a.model)
    files = ":".join(str(f) for f in ix.gguf.files)
    device = device_for(ix.gguf.files[0])
    members = member_devices(device)
    prompt = a.prompt.read_text()
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.label or a.model.stem[:24]
    results: list[dict] = []
    sock = f"/tmp/tierinfer-{os.getpid()}.sock"

    for rep in range(a.first_rep - 1, a.first_rep - 1 + a.repeat):
        for arm in a.arms.split(","):
            label = f"{stem}-{arm}-{rep + 1}"
            print(f"== {label}")
            r = drop_cache(a.model)
            print(f"   cold: {r.fraction:.2%} resident after drop")
            server = None
            loader = None
            if arm == "loader":
                if not SHIM.exists():
                    print(f"shim not built at {SHIM}", file=sys.stderr)
                    return 2
                tel = a.out / f"{label}.telemetry.jsonl"
                cmd = [sys.executable, "-m", "tierinfer.cli", "serve", str(a.model), "--sock", sock,
                       "--workers", str(a.loader_workers), "--depth", str(a.loader_depth), "--telemetry", str(tel)]
                if a.loader_ram_gb:
                    cmd += ["--ram-gb", str(a.loader_ram_gb)]
                env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
                server = subprocess.Popen(cmd, env=env, stdout=open(a.out / f"{label}.server.log", "w"),
                                          stderr=subprocess.STDOUT)
                # Wait until the server *accepts*, not until the path exists: a
                # stale socket file from the previous repetition exists before
                # the new server has bound, and llama-server started in that gap
                # found nobody home and ran native — two of three GLM "loader"
                # repetitions did exactly that on 2026-09-18.
                import socket as _socket
                up = False
                for _ in range(600):
                    try:
                        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as probe:
                            probe.settimeout(0.5)
                            probe.connect(sock)
                            probe.sendall(b"HELLO probe 0\n")
                        up = True
                        break
                    except OSError:
                        time.sleep(0.1)
                    if server.poll() is not None:
                        break
                if not up:
                    print(f"tierinfer serve did not accept on {sock}; see {a.out / (label + '.server.log')}",
                          file=sys.stderr)
                    return 2
                loader = {"sock": sock, "files": files}
            try:
                res = run_arm(label, a.model, prompt, a.n_predict, a.extra, port=a.port, device=device,
                              members=members, out=a.out, loader=loader,
                              memory_max=a.native_memory_max if arm == "native" else None,
                              server_log=a.out / f"{label}.llama.log")
            finally:
                if server is not None:
                    time.sleep(1)
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
            if arm == "loader":
                llama_log = (a.out / f"{label}.llama.log").read_text(errors="replace")
                if "standing aside" in llama_log or "through userfaultfd" not in llama_log:
                    res["stood_aside"] = True
                    print("   WARNING: the shim stood aside — this run is NATIVE, not a loader run", file=sys.stderr)
                # the last snapshot of the server's telemetry, for the table
                try:
                    from tierinfer.telemetry import read
                    recs = read(a.out / f"{label}.telemetry.jsonl")
                    snaps = [r for r in recs if r["type"] == "snapshot"]
                    close = [r for r in recs if r.get("kind") == "run.close"]
                    res["loader"] = (close[-1] if close else (snaps[-1]["values"] if snaps else {}))
                except Exception as e:  # noqa: BLE001 — telemetry missing is reported, not fatal
                    res["loader"] = {"error": str(e)}
            res["arm"] = arm
            res["rep"] = rep + 1
            results.append(res)
            (a.out / f"{label}.json").write_text(json.dumps(res, indent=1))
            print(f"   load {res['load_s']:.0f}s  prompt {res['prompt_tps']:.3f} t/s  gen {res['gen_tps']:.3f} t/s  "
                  f"infer io {res['io_infer']['bytes'] / GB:.2f} GB / {res['io_infer']['reads']} reads @ "
                  f"{res['io_infer']['mean_read_bytes'] / 1024:.0f} KB")

    # -- summary --------------------------------------------------------
    by_arm: dict[str, list[dict]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)
    natives = by_arm.get("native", [])
    loaders = by_arm.get("loader", [])
    same = None
    if natives and loaders:
        same = all(l["content"] == natives[0]["content"] for l in loaders) and \
            all(n["content"] == natives[0]["content"] for n in natives)
    lines = [f"# {stem}: native vs loader ({a.repeat} runs each, n_predict {a.n_predict}, extra {' '.join(a.extra)})", "",
             "| arm | run | load s | prompt t/s | gen t/s | infer GB | infer reads | mean KB | await ms | RSS GB | tokens identical to native #1 |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for r in results:
        io = r["io_infer"]
        ident = "—" if not natives else ("yes" if r["content"] == natives[0]["content"] else "**NO**")
        if r.get("stood_aside"):
            ident += " (SHIM STOOD ASIDE — native run)"
        lines.append(f"| {r['arm']} | {r['rep']} | {r['load_s']:.0f} | {r['prompt_tps']:.3f} | {r['gen_tps']:.3f} | "
                     f"{io['bytes'] / GB:.2f} | {io['reads']} | {io['mean_read_bytes'] / 1024:.0f} | {io['await_ms']:.2f} | "
                     f"{r['server'].get('VmHWM', 0) / GB:.0f} | {ident} |")
    for arm, rs in by_arm.items():
        if len(rs) > 1:
            g = [r["gen_tps"] for r in rs]
            lines.append(f"| **{arm} median** | | | | **{statistics.median(g):.3f}** ({min(g):.3f}–{max(g):.3f}) | "
                         f"{statistics.median(r['io_infer']['bytes'] for r in rs) / GB:.2f} | "
                         f"{statistics.median(r['io_infer']['reads'] for r in rs):.0f} | | | | |")
    if loaders:
        lines += ["", "| loader run | hits | misses | hit rate | faults | copied GB | evictions | prefetch issued/useful/late/wasted |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for r in loaders:
            L = r.get("loader", {})
            h, m = L.get("loader.hits", 0), L.get("loader.misses", 0)
            lines.append(f"| {r['rep']} | {h} | {m} | {h / (h + m) if h + m else 0:.1%} | {L.get('loader.faults', 0)} | "
                         f"{L.get('loader.bytes_copied', 0) / GB:.1f} | {L.get('loader.evictions', 0)} | "
                         f"{L.get('loader.prefetch_issued', 0)}/{L.get('loader.prefetch_useful', 0)}/"
                         f"{L.get('loader.prefetch_late', 0)}/{L.get('loader.prefetch_wasted', 0)} |")
    lines += ["", f"Greedy tokens identical across all runs and arms: **{same}**" if same is not None else ""]
    (a.out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

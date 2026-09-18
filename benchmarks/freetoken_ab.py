"""FreeToken with and without a TierInfer tier under its CPU-executor layers.

Two arms, same `ft serve` flags, same prompt, greedy, cold page cache:

  native  — FreeToken as it is: every bank read into RAM at load; the
            `--moe-cpu-layers` layers decode on the CPU executor from their
            resident banks.
  tiered  — the same layers' banks are TierInfer-served buffers
            (`HostResidency.TIERED`): rows arrive on first touch from the FTW
            shards and leave under `--tier-gb`, chosen below those layers' size.

Each run records FreeToken's own numbers (`/v1/stats` before and after, the
completion's usage and wall time), TierInfer's telemetry for the tiered arm,
and the block device's counters around inference. Greedy tokens are compared
across arms. The FreeToken checkout must carry the `tierinfer-tier` patch.

    PYTHONPATH=src python benchmarks/freetoken_ab.py /data/ai-data/models/<ftw-dir> \\
        --cpu-layers 12 --tier-gb 8 --repeat 3 --n-predict 64
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tierinfer.bench import device_for, disk_counters, member_devices, residency_of  # noqa: E402

FT_ROOT = Path(os.environ.get("FREETOKEN_ROOT", Path.home() / "freetoken-qwen38"))
FT = FT_ROOT / ".venv" / "bin" / "ft"
CUDA = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")

PROMPT = ("Design an expert cache for a sparse mixture-of-experts model whose weights live on "
          "NVMe RAID. Cover the cache policy, prefetching on routing, and how to measure whether "
          "it helps. Be concrete and technical.")


def _drop(paths: list[Path]) -> float:
    """posix_fadvise DONTNEED on every shard; returns the resident fraction after (mincore)."""
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return residency_of(paths).fraction


def _snap(device: str, members: list[str]):
    return disk_counters(device), {m: disk_counters(m) for m in members}


def _delta(a, b, members: list[str]) -> dict:
    d = b[0] - a[0]
    return {"reads": d.reads, "bytes": d.bytes_read, "mean_read_bytes": d.mean_read_bytes, "await_ms": d.await_ms,
            "members": {m: {"reads": (b[1][m] - a[1][m]).reads, "bytes": (b[1][m] - a[1][m]).bytes_read,
                            "mean_read_bytes": (b[1][m] - a[1][m]).mean_read_bytes} for m in members}}


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _wait_health(port: int, proc: subprocess.Popen, timeout: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"ft serve exited with {proc.returncode} before it was healthy")
        try:
            h = _get(f"http://127.0.0.1:{port}/health", timeout=2)
            if h.get("status") == "ok":                 # loading -> ok -> error (FreeToken's own lifecycle)
                return time.time() - t0
            if h.get("status") == "error":
                raise RuntimeError(f"ft serve reports error: {h.get('message')}")
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"ft serve not healthy after {timeout:.0f} s")


def _complete(port: int, n_predict: int) -> tuple[dict, float]:
    body = json.dumps({"model": "ab", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": n_predict, "temperature": 0.0, "top_p": 1.0, "seed": 1,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        doc = json.loads(r.read().decode())
    return doc, time.perf_counter() - t0


def _stats(port: int) -> dict:
    try:
        return _get(f"http://127.0.0.1:{port}/v1/stats", timeout=5)
    except Exception as exc:  # the stats surface is FreeToken's; its absence is recorded, not fatal
        return {"error": str(exc)}


def _kill(proc: subprocess.Popen | None, name: str, grace: float = 20.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print(f"   {name} stopped ({proc.returncode})")


def run_arm(label: str, arm: str, a, out: Path, device: str, members: list[str]) -> dict:
    shards = sorted(a.model.glob("*.ftw"))
    frac = _drop(shards)
    print(f"== {label}\n   cold: {frac:.2%} resident after drop", flush=True)
    env = dict(os.environ, CUDA_HOME=CUDA, PYTHONUNBUFFERED="1",
               PATH=f"{CUDA}/bin:{FT_ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}",
               LD_LIBRARY_PATH=f"{CUDA}/lib64")
    env.pop("TIERINFER_SOCK", None)
    server = None
    sock = None
    tel = out / f"{label}.telemetry.jsonl"
    if arm == "tiered":
        sock = f"/tmp/tierinfer-ft-{os.getpid()}.sock"
        try:
            os.unlink(sock)
        except FileNotFoundError:
            pass
        cmd = [sys.executable, "-m", "tierinfer.cli", "serve", str(a.model), "--sock", sock,
               "--ram-gb", str(a.tier_gb), "--workers", str(a.workers), "--depth", str(a.depth),
               "--telemetry", str(tel)]
        server = subprocess.Popen(cmd, stdout=open(out / f"{label}.server.log", "w"), stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        for _ in range(100):
            if os.path.exists(sock):
                break
            time.sleep(0.1)
        else:
            _kill(server, "tierinfer serve")
            raise RuntimeError("tierinfer serve did not come up")
        env["TIERINFER_SOCK"] = sock
        env["TIERINFER_SRC"] = str(ROOT / "src")
    ft_cmd = [str(FT), "serve", "--model", str(a.model), "--served-model-name", "ab",
              "--host", "127.0.0.1", "--port", str(a.port), "--gpu", "0",
              "--moe-strategy", "offload", "--moe-cpu-layers", a.cpu_layers, *a.extra]
    io0 = _snap(device, members)
    t0 = time.time()
    ft = subprocess.Popen(ft_cmd, cwd=str(FT_ROOT), env=env,
                          stdout=open(out / f"{label}.ft.log", "w"), stderr=subprocess.STDOUT)
    try:
        load_s = _wait_health(a.port, ft, a.load_timeout)
        io_load = _snap(device, members)
        stats0 = _stats(a.port)
        killer = None
        if server is not None and a.kill_server_after > 0:
            import threading

            def _kill_server():
                time.sleep(a.kill_server_after)
                print(f"   injecting: SIGKILL tierinfer serve (pid {server.pid}) {a.kill_server_after:.1f} s into the completion", flush=True)
                server.kill()

            killer = threading.Thread(target=_kill_server, daemon=True)
            killer.start()
        doc, wall = _complete(a.port, a.n_predict)
        stats1 = _stats(a.port)
        if killer is not None:
            killer.join(5)
        io1 = _snap(device, members)
    finally:
        _kill(ft, "ft serve", grace=30)
        if server is not None:
            _kill(server, "tierinfer serve")
            if sock:
                try:
                    os.unlink(sock)
                except FileNotFoundError:
                    pass
    choice = (doc.get("choices") or [{}])[0].get("message", {})
    text = (choice.get("reasoning_content") or "") + "␟" + (choice.get("content") or "")
    usage = doc.get("usage", {})
    res = {"label": label, "cmd": ft_cmd, "load_s": load_s, "wall_s": wall,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
           "text": text, "stats_before": stats0, "stats_after": stats1,
           "io_load": _delta(io0, io_load, members), "io_infer": _delta(io_load, io1, members),
           "arm": arm, "rep": label.rsplit("-", 1)[-1], "depth": a.depth if arm == "tiered" else None,
           "tier_gb": a.tier_gb if arm == "tiered" else None, "cpu_layers": a.cpu_layers,
           "killed_server_after_s": a.kill_server_after or None}
    if usage.get("completion_tokens"):
        res["gen_tps_wall"] = usage["completion_tokens"] / wall     # upper bound on decode time: includes prefill
    (out / f"{label}.json").write_text(json.dumps(res, indent=1))
    ii = res["io_infer"]
    print(f"   load {load_s:.0f}s  wall {wall:.1f}s  completion {usage.get('completion_tokens')} tok "
          f"({res.get('gen_tps_wall', 0):.3f} tok/s incl. prefill)  infer io {ii['bytes'] / 1e9:.2f} GB / "
          f"{ii['reads']} reads @ {ii.get('mean_read_bytes', 0) / 1024:.0f} KB", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="FTW checkpoint directory")
    ap.add_argument("--cpu-layers", default="12", help="--moe-cpu-layers spec (count, fraction or ids)")
    ap.add_argument("--tier-gb", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--arms", default="native,tiered")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--load-timeout", type=float, default=1800)
    ap.add_argument("--label", default="flashnext")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "freetoken-out")
    ap.add_argument("--kill-server-after", type=float, default=0.0,
                    help="failure injection: SIGKILL `tierinfer serve` this many seconds into the completion (tiered arm)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more ft serve args (after --extra)")
    a = ap.parse_args()
    if not FT.exists():
        print(f"no ft at {FT}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    device = device_for(a.model)
    members = member_devices(device)
    results = []
    for rep in range(a.first_rep - 1, a.first_rep - 1 + a.repeat):
        for arm in a.arms.split(","):
            results.append(run_arm(f"{a.label}-{arm}-{rep + 1}", arm, a, a.out, device, members))
    texts = {r["text"] for r in results}
    print(f"\nGreedy output identical across runs and arms: {len(texts) == 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

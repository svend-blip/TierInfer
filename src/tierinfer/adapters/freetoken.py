"""FreeToken: give it a budget, take its numbers back in one shape.

FreeToken is an MoE serving engine that already does what goals 5, 9 and 10
of this project do — a global LRU expert cache, elastic VRAM re-allocation
between that cache and KV memory, CPU–GPU co-execution. It owns its loading
path, and goal 6 measured what happens to anyone who tries to manage
residency from outside a runtime that owns its own: nothing happens, slowly.

So this adapter does the two things that are left, and they are the two
things TierInfer is actually better placed to do.

**Configuration in.** FreeToken takes an expert cache size, a KV reservation
and a memory ratio as launch flags. Those are exactly the numbers
`autoconfig.configure` derives from the card, the kernel and the model's own
metadata — including the refusals, so a context that cannot work is caught
before a server starts rather than after it has loaded 56 GB.

**Telemetry out.** `/v1/stats` reports KV pages, VRAM bytes and throughput in
FreeToken's own shape. `normalise` puts them in the `runtime.*` namespace
that every adapter here produces, which is the only way the schema's claim to
be runtime-independent can be checked rather than asserted.

Nothing here talks to a GPU or moves a weight. The HTTP client is
`urllib.request` from the standard library: an adapter that needed a
dependency to read a JSON document would be a poor trade.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from ..autoconfig import Configuration
from ..telemetry import TelemetryError
from ..vram import GB, MB


class FreeTokenError(RuntimeError):
    pass


class FreeTokenUnreachable(FreeTokenError):
    """The server is not answering. A fact about the host, not a bug."""


#: What every adapter in this package produces, so two runtimes can be put in
#: one table. Declared here and asserted by the tests rather than left to
#: each adapter to remember.
RUNTIME_FIELDS = (
    "runtime.decode_tps", "runtime.prefill_tps", "runtime.kv_used_pages",
    "runtime.kv_total_pages", "runtime.kv_page_size", "runtime.vram_bytes",
    "runtime.requests_active", "runtime.requests_completed", "runtime.uptime_s",
)


# -- configuration in ---------------------------------------------------


def launch_arguments(config: Configuration, *,
                     policy: str = "lru", memory_ratio: float | None = None
                     ) -> list[str]:
    """FreeToken launch flags for a configuration TierInfer derived.

    Raises when the configuration is not usable. FreeToken would start
    anyway and discover the problem after loading the model, which on this
    model is 56 GB and forty seconds of finding out something already known.
    """
    if not config.usable:
        raise FreeTokenError(
            "this configuration cannot work and FreeToken would only discover "
            "that after loading the model: " + "; ".join(config.problems))

    cache_bytes = config.vram_bytes or config.ram_bytes
    if cache_bytes <= 0:
        raise FreeTokenError("the configuration leaves no room for an expert cache")

    args = ["--moe-cache-size", f"{cache_bytes // MB}M",
            "--moe-cache-policy", policy,
            "--kv-reserve-tokens", str(config.context_length)]
    if memory_ratio is not None:
        if not 0.0 < memory_ratio <= 1.0:
            raise FreeTokenError("memory ratio must be in (0, 1]")
        args += ["--memory-ratio", f"{memory_ratio:g}"]
    return args


def describe_launch(config: Configuration, **kw: Any) -> str:
    """The flags, and what each one was derived from."""
    args = launch_arguments(config, **kw)
    pairs = dict(zip(args[::2], args[1::2]))
    lines = [f"  {k} {v}" for k, v in pairs.items()]
    why = [f"  cache from {'VRAM' if config.vram_bytes else 'RAM'} budget: "
           f"{(config.vram_bytes or config.ram_bytes) / GB:.1f} GB",
           f"  KV reservation from the context asked for: {config.context_length}"]
    return "flags:\n" + "\n".join(lines) + "\nderived from:\n" + "\n".join(why)


# -- telemetry out ------------------------------------------------------


def normalise(stats: Mapping[str, Any]) -> dict[str, Any]:
    """FreeToken's `/v1/stats` document in the shared `runtime.*` namespace.

    Absent fields come back as ``None`` rather than zero. A runtime that does
    not report a counter has not reported zero of it, and the difference
    matters the moment two runtimes are compared.
    """
    if not isinstance(stats, Mapping):
        raise FreeTokenError(f"stats must be a mapping, got {type(stats).__name__}")
    kv = stats.get("kv") or {}
    throughput = stats.get("throughput") or {}
    requests = stats.get("requests") or {}
    return {
        "runtime.decode_tps": throughput.get("decode_tps"),
        "runtime.prefill_tps": throughput.get("prefill_tps"),
        "runtime.kv_used_pages": kv.get("used_pages"),
        "runtime.kv_total_pages": kv.get("total_pages"),
        "runtime.kv_page_size": kv.get("page_size"),
        "runtime.vram_bytes": stats.get("vram_bytes"),
        "runtime.requests_active": requests.get("active"),
        "runtime.requests_completed": requests.get("completed"),
        "runtime.uptime_s": stats.get("uptime_s"),
    }


@dataclass
class FreeTokenClient:
    """Reads a running FreeToken. Read-only by construction."""

    base_url: str = "http://127.0.0.1:8080"
    timeout: float = 5.0

    def _get(self, path: str) -> Any:
        url = self.base_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise FreeTokenUnreachable(f"{url}: {e.reason}") from None
        except (OSError, json.JSONDecodeError) as e:
            raise FreeTokenError(f"{url}: {e}") from None

    def health(self) -> dict:
        return self._get("/health")

    def stats(self) -> dict:
        return self._get("/v1/stats")

    def snapshot_values(self) -> dict[str, Any]:
        """Normalised counters, ready for `Telemetry.snapshot`."""
        return normalise(self.stats())

    def is_up(self) -> bool:
        try:
            self.health()
            return True
        except FreeTokenError:
            return False

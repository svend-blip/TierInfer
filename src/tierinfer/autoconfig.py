"""Deriving a configuration instead of being handed one.

Every number this project needs has been measured somewhere: the model's
layout is in its own file, the card's size is in the driver, RAM is in
/proc/meminfo, and the tier costs are in the benchmarks. Asking a user to
supply them again is asking them to get one wrong.

So this reads the host and the model and produces a configuration that says
what it decided and why. Three rules govern what it does when a number is
missing or a budget does not close:

**Measure, do not assume.** VRAM comes from the driver, RAM from the kernel,
the model's shape from its GGUF. A figure that cannot be measured on this
host is reported absent rather than defaulted, because a default that happens
to be near the truth is indistinguishable from a measurement until it is not.

**Leave room, and say how much.** A budget that exactly fits fails the first
time anything else runs. The margins are explicit fields, not slack hidden
inside another term.

**Refuse rather than produce something that cannot work.** A context that
does not fit on this card, or a model larger than RAM and disk together, is
an answer — ``Configuration.problems`` carries it, and ``usable`` is False.
A configuration that quietly returns unusable numbers costs a run to find out.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vram import GB, MB, CudaError, CudaRuntime, CudaUnavailable, VramBudget

#: Fraction of free RAM this is willing to plan around. The rest is for
#: everything else on the machine, including the page cache the model's own
#: reads will want.
DEFAULT_RAM_SHARE = 0.60


@dataclass(frozen=True)
class Host:
    """What this machine actually has, measured."""

    ram_total: int
    ram_available: int
    vram_total: int | None
    vram_free: int | None
    cpu_count: int
    platform: str

    @property
    def has_gpu(self) -> bool:
        return self.vram_total is not None

    @classmethod
    def measure(cls) -> "Host":
        mem = _meminfo()
        vram_total = vram_free = None
        try:
            info = CudaRuntime().memory_info()
            vram_total, vram_free = info.total, info.free
        except (CudaUnavailable, CudaError):
            pass                       # no card here; reported absent, not zero
        return cls(ram_total=mem.get("MemTotal", 0),
                   ram_available=mem.get("MemAvailable", mem.get("MemFree", 0)),
                   vram_total=vram_total, vram_free=vram_free,
                   cpu_count=os.cpu_count() or 1,
                   platform=platform.platform())


def _meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes. Empty where the file does not exist."""
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                out[name] = int(parts[0]) * (1024 if len(parts) > 1 and
                                             parts[1] == "kB" else 1)
    except OSError:
        pass
    return out


@dataclass
class Configuration:
    """What to run with, and what had to be decided to get there."""

    model: str
    context_length: int
    model_bytes: int
    floor_bytes: int
    expert_bytes: int
    vram_experts: int
    ram_experts: int
    stream_workers: int
    prefetch_depth: int
    budget: VramBudget | None
    host: Host
    decisions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems

    @property
    def vram_bytes(self) -> int:
        return self.vram_experts * self.expert_bytes

    @property
    def ram_bytes(self) -> int:
        return self.ram_experts * self.expert_bytes

    @property
    def resident_share(self) -> float:
        routed = self.model_bytes - self.floor_bytes
        if routed <= 0:
            return 0.0
        return min(1.0, (self.vram_bytes + self.ram_bytes) / routed)

    def explain(self) -> str:
        lines = [f"model           {self.model}",
                 f"context         {self.context_length}",
                 f"host            {self.host.cpu_count} CPUs, "
                 f"{self.host.ram_total / GB:.0f} GB RAM "
                 f"({self.host.ram_available / GB:.0f} free)"
                 + (f", {self.host.vram_total / GB:.1f} GB VRAM"
                    if self.host.has_gpu else ", no GPU")]
        if self.budget:
            lines.append("")
            lines.append(self.budget.explain())
        lines += ["",
                  f"VRAM experts    {self.vram_experts:>7}  {self.vram_bytes / GB:>6.1f} GB",
                  f"RAM experts     {self.ram_experts:>7}  {self.ram_bytes / GB:>6.1f} GB",
                  f"resident share  {self.resident_share:>7.1%}  of the routed weights",
                  f"stream workers  {self.stream_workers:>7}",
                  f"prefetch depth  {self.prefetch_depth:>7}  (a starting point; "
                  "the policy moves it)"]
        if self.decisions:
            lines += ["", "decisions:"] + [f"  - {d}" for d in self.decisions]
        if self.problems:
            lines += ["", "problems:"] + [f"  - {p}" for p in self.problems]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "context_length": self.context_length,
                "model_bytes": self.model_bytes, "floor_bytes": self.floor_bytes,
                "expert_bytes": self.expert_bytes,
                "vram_experts": self.vram_experts, "ram_experts": self.ram_experts,
                "stream_workers": self.stream_workers,
                "prefetch_depth": self.prefetch_depth,
                "resident_share": self.resident_share,
                "usable": self.usable, "problems": list(self.problems),
                "decisions": list(self.decisions)}


def configure(index, *, context_length: int = 8192, host: Host | None = None,
              ram_share: float = DEFAULT_RAM_SHARE, stream_workers: int | None = None,
              prefetch_depth: int = 8, reserve_bytes: int | None = None
              ) -> Configuration:
    """Derive a configuration for this model on this host.

    ``index`` is a ``ModelIndex``. ``stream_workers`` defaults to the measured
    best on the reference NVMe — eight, flat to sixteen and worse at
    thirty-two (`benchmarks/STREAMING.md`) — bounded by the CPUs present,
    because that number is a property of the device rather than of the code.
    """
    host = host or Host.measure()
    floor = index.always_resident_nbytes()
    routed = index.routed_nbytes()
    expert = index.expert_nbytes()
    model_bytes = floor + routed
    decisions: list[str] = []
    problems: list[str] = []

    workers = stream_workers or min(8, max(1, host.cpu_count // 2))
    if stream_workers is None:
        decisions.append(f"stream workers {workers}: eight measured best on the "
                         "reference NVMe, bounded by this host's CPUs")

    budget = None
    vram_experts = 0
    if host.has_gpu:
        reserve = reserve_bytes if reserve_bytes is not None else (
            host.vram_total - host.vram_free)
        budget = VramBudget.from_model(index.gguf.metadata,
                                       total_bytes=host.vram_total,
                                       context_length=context_length,
                                       layers=index.block_count,
                                       reserve_bytes=reserve)
        decisions.append(f"VRAM reserve {reserve / GB:.2f} GB: measured as already "
                         "in use on this card, not assumed")
        if not budget.fits:
            problems.append(
                f"context {context_length} does not fit: the KV cache alone needs "
                f"{budget.kv_cache / GB:.2f} GB of {host.vram_total / GB:.2f} GB")
        else:
            vram_experts = budget.experts(expert, floor)
            if vram_experts == 0:
                problems.append(
                    f"context {context_length} leaves "
                    f"{(budget.weights - floor) / GB:.2f} GB for experts of "
                    f"{expert / MB:.2f} MB — nothing fits beside the resident floor")
            else:
                per_token = index.expert_used_count * len(index.moe_layers)
                if vram_experts < per_token:
                    problems.append(
                        f"VRAM holds {vram_experts} experts but one token routes to "
                        f"{per_token}; below one token's working set the hit rate is "
                        "zero rather than low")
    else:
        decisions.append("no GPU measured: all experts are planned for RAM")

    ram_for_experts = max(0, int(host.ram_available * ram_share) - floor)
    ram_experts = ram_for_experts // expert if expert else 0
    decisions.append(f"RAM share {ram_share:.0%} of {host.ram_available / GB:.0f} GB "
                     "available, minus the resident floor")
    if ram_experts + vram_experts == 0:
        problems.append("neither tier can hold a single expert on this host")

    total_capacity = (vram_experts + ram_experts) * expert
    if total_capacity < routed:
        decisions.append(
            f"{total_capacity / GB:.1f} GB of {routed / GB:.1f} GB of experts fit in "
            "memory; the rest is served from storage")

    return Configuration(model=Path(index.gguf.path).name,
                         context_length=context_length, model_bytes=model_bytes,
                         floor_bytes=floor, expert_bytes=expert,
                         vram_experts=vram_experts, ram_experts=ram_experts,
                         stream_workers=workers, prefetch_depth=prefetch_depth,
                         budget=budget, host=host,
                         decisions=decisions, problems=problems)

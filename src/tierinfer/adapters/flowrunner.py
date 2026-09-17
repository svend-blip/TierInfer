"""FlowRunner: a capability declared in a file, and telemetry back out.

FlowRunner runs work as flows built from declared steps, where each step
names what it needs and the runner supplies it. For TierInfer to be usable
there it has to be describable in a file and reportable in a machine-readable
shape — the same two halves as the FreeToken adapter, with the declaration
coming in rather than the flags going out.

**A capability declaration is a request, not a configuration.** It says what
the work wants — this model, this context, an optional floor under residency
— and is deliberately unable to say how much VRAM to use, because that is a
property of the host the flow lands on and not of the flow. `resolve` puts
the two together and produces a configuration, or a refusal naming what the
host cannot give.

That split is the point. A flow that declared 22 GB of expert cache would run
on one machine and fail on the next; a flow that declares a context and a
model runs wherever the numbers work out, and says plainly where they do not.

The format is a small JSON document, validated on the way in. Unknown keys
are refused rather than ignored: a misspelled field that is silently dropped
is a setting that does not apply, discovered later and expensively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..autoconfig import Configuration, Host, configure
from ..vram import GB, MB

CAPABILITY_VERSION = 1
CAPABILITY_NAME = "tierinfer.residency"

_REQUIRED = ("version", "capability", "model")
_OPTIONAL = ("context_length", "min_resident_share", "ram_share",
             "stream_workers", "prefetch_depth", "notes")
#: Things a flow must not name, because they belong to whatever machine the
#: flow lands on. Declared separately so the error can say why.
_HOST_PROPERTIES = ("vram_bytes", "vram_gb", "cache_size", "moe_cache_size",
                    "ram_bytes", "ram_gb", "moe_cache_rate")


class CapabilityError(ValueError):
    pass


@dataclass(frozen=True)
class Capability:
    """What a flow step asks for. Host-independent by construction."""

    model: str
    context_length: int = 8192
    #: Refuse the step rather than run it if less than this share of the
    #: routed weights would be held in memory. A flow that only makes sense
    #: at speed can say so instead of running slowly and being believed.
    min_resident_share: float | None = None
    ram_share: float | None = None
    stream_workers: int | None = None
    prefetch_depth: int = 8
    notes: str = ""

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "Capability":
        if not isinstance(doc, Mapping):
            raise CapabilityError(f"a capability is an object, not {type(doc).__name__}")
        missing = [k for k in _REQUIRED if k not in doc]
        if missing:
            raise CapabilityError(f"missing required field(s): {', '.join(missing)}")
        # The host-property check runs before the unknown-field check, so a
        # flow naming a VRAM size is told why it may not rather than merely
        # that the key is unrecognised. The specific error is the useful one.
        for forbidden in _HOST_PROPERTIES:
            if forbidden in doc:
                raise CapabilityError(
                    f"{forbidden!r} is a property of the host, not of the flow; "
                    "declare a context and let it be derived")
        unknown = set(doc) - set(_REQUIRED) - set(_OPTIONAL)
        if unknown:
            raise CapabilityError(
                f"unknown field(s): {', '.join(sorted(unknown))} — refused rather "
                "than ignored, because a silently dropped setting is one that does "
                "not apply and is found out later")
        if doc["version"] != CAPABILITY_VERSION:
            raise CapabilityError(
                f"capability version {doc['version']}, this reader knows "
                f"{CAPABILITY_VERSION}")
        if doc["capability"] != CAPABILITY_NAME:
            raise CapabilityError(
                f"capability {doc['capability']!r}, this reader provides "
                f"{CAPABILITY_NAME!r}")
        ctx = int(doc.get("context_length", 8192))
        if ctx <= 0:
            raise CapabilityError("context_length must be positive")
        share = doc.get("min_resident_share")
        if share is not None and not 0.0 <= float(share) <= 1.0:
            raise CapabilityError("min_resident_share must be between 0 and 1")
        return cls(model=str(doc["model"]), context_length=ctx,
                   min_resident_share=None if share is None else float(share),
                   ram_share=(None if doc.get("ram_share") is None
                              else float(doc["ram_share"])),
                   stream_workers=(None if doc.get("stream_workers") is None
                                   else int(doc["stream_workers"])),
                   prefetch_depth=int(doc.get("prefetch_depth", 8)),
                   notes=str(doc.get("notes", "")))

    @classmethod
    def load(cls, path: str | Path) -> "Capability":
        try:
            doc = json.loads(Path(path).read_text())
        except json.JSONDecodeError as e:
            raise CapabilityError(f"{path}: not JSON ({e})") from None
        except OSError as e:
            raise CapabilityError(f"{path}: {e}") from None
        return cls.from_dict(doc)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": CAPABILITY_VERSION,
                               "capability": CAPABILITY_NAME,
                               "model": self.model,
                               "context_length": self.context_length,
                               "prefetch_depth": self.prefetch_depth}
        for name in ("min_resident_share", "ram_share", "stream_workers"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.notes:
            out["notes"] = self.notes
        return out


@dataclass
class Resolution:
    """What the host can actually give, against what the flow asked for."""

    capability: Capability
    configuration: Configuration | None
    refusals: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.configuration is not None and not self.refusals

    def explain(self) -> str:
        lines = [f"capability      {CAPABILITY_NAME} v{CAPABILITY_VERSION}",
                 f"asked for       {self.capability.model} @ "
                 f"{self.capability.context_length} tokens"]
        if self.capability.min_resident_share is not None:
            lines.append(f"requires        {self.capability.min_resident_share:.0%} "
                         "of routed weights in memory")
        if self.configuration is not None:
            lines += ["", self.configuration.explain()]
        if self.refusals:
            lines += ["", "not available here:"] + [f"  - {r}" for r in self.refusals]
        return "\n".join(lines)


def resolve(capability: Capability, index, *, host: Host | None = None) -> Resolution:
    """Put a declaration and a host together, or say why they do not fit."""
    kwargs: dict[str, Any] = {"context_length": capability.context_length,
                              "prefetch_depth": capability.prefetch_depth,
                              "host": host}
    if capability.ram_share is not None:
        kwargs["ram_share"] = capability.ram_share
    if capability.stream_workers is not None:
        kwargs["stream_workers"] = capability.stream_workers
    config = configure(index, **kwargs)
    refusals = list(config.problems)
    need = capability.min_resident_share
    if need is not None and config.resident_share < need:
        refusals.append(
            f"this host holds {config.resident_share:.0%} of the routed weights, "
            f"below the {need:.0%} the step declared it needs")
    return Resolution(capability=capability, configuration=config, refusals=refusals)


def telemetry_values(config: Configuration) -> dict[str, Any]:
    """What a resolved step contributes to a run's telemetry, namespaced."""
    return {
        "capability.context_length": config.context_length,
        "capability.vram_experts": config.vram_experts,
        "capability.ram_experts": config.ram_experts,
        "capability.resident_share": round(config.resident_share, 6),
        "capability.stream_workers": config.stream_workers,
        "capability.prefetch_depth": config.prefetch_depth,
        "capability.usable": config.usable,
    }

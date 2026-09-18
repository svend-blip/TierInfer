"""One record of what happened, whatever was running underneath.

Every component here already counts what it does: the tracker counts
activations, the caches count hits, the storage backend counts bytes and
seconds, the policy counts stalls. Those counters are useful while a
benchmark is printing a table and useless afterwards, because each has its
own shape and none of them says which run it came from.

This is the shape they share. It is deliberately small and deliberately
runtime-independent: a snapshot taken behind llama.cpp, behind FreeToken or
behind a simulation must be comparable to one taken behind any other, or the
numbers cannot be put in the same table — which is the only reason to collect
them.

Two kinds of record:

``Event``     something happened at a moment — a run opened, a policy moved
              its depth, a tier was exhausted
``Snapshot``  what the counters read at a moment, flattened into one
              namespaced mapping

Both carry the schema version, so a file written today can be read by
something that knows it was written before a field existed. They are written
as JSON Lines, one record per line, appended: a reader that arrives halfway
through a run gets everything up to that point and nothing that has not
happened yet.

**Nothing here computes.** A metric that is derived on the way out cannot be
checked against the thing it was derived from. Rates and ratios belong to
whoever reads the file.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SCHEMA_VERSION = 1


class TelemetryError(RuntimeError):
    pass


def _scalar(value: Any) -> Any:
    """What may appear in a record: numbers, strings, booleans, null.

    Nested structure is refused rather than flattened silently, because a
    reader that has to guess how a nested field was spelled is a reader that
    will guess wrong.
    """
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TelemetryError(f"telemetry values must be scalar, got {type(value).__name__}")


@dataclass(frozen=True)
class Event:
    """Something that happened, at a moment."""

    kind: str
    at: float
    run: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"v": SCHEMA_VERSION, "type": "event", "kind": self.kind,
                           "at": round(self.at, 6), "run": self.run,
                           **{k: _scalar(v) for k, v in self.fields.items()}})


@dataclass(frozen=True)
class Snapshot:
    """What the counters read, at a moment, namespaced by component."""

    at: float
    run: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"v": SCHEMA_VERSION, "type": "snapshot",
                           "at": round(self.at, 6), "run": self.run,
                           "values": {k: _scalar(v) for k, v in sorted(self.values.items())}})


# -- collecting ---------------------------------------------------------


#: What a component is asked for. Anything with these attributes can be
#: collected without this module importing it, which is what keeps the
#: schema runtime-independent rather than a union of everything that exists.
def collect(namespace: str, source: Any, fields: Iterable[str]) -> dict[str, Any]:
    """Read named counters off a component into a namespaced mapping.

    A missing field is an error rather than a gap: a snapshot with a silently
    absent column reads as a zero later, and a zero that was never measured
    is the most expensive kind of wrong number.
    """
    out: dict[str, Any] = {}
    for name in fields:
        if not hasattr(source, name):
            raise TelemetryError(f"{type(source).__name__} has no {name!r} to collect")
        value = getattr(source, name)
        out[f"{namespace}.{name}"] = _scalar(value() if callable(value) else value)
    return out


#: The fields each component contributes. Named here rather than in the
#: components so that a schema change is one edit in one place, and so that
#: adding a counter somewhere does not silently widen the schema.
STANDARD_FIELDS: dict[str, tuple[str, ...]] = {
    "storage": ("bytes_read", "operations", "seconds"),
    "stream": ("submitted", "completed", "failed", "cancelled", "bytes_read",
               "read_seconds", "queue_seconds"),
    "cache": ("hits", "misses", "insertions", "evictions", "bytes_admitted",
              "bytes_evicted"),
    "vram": ("hits", "misses", "transfers", "evictions", "bytes_transferred",
             "transfer_seconds", "refused"),
    "prefetch": ("issued", "used", "late", "cancelled", "stalls", "stalls_avoided",
                 "bytes_issued", "wasted_bytes", "exact_fallbacks", "late_wait_seconds",
                 "timed_out"),
    #: Counters of the policy *simulator* (`tierinfer.policy`). Its ``seconds``
    #: are computed from constants, not measured, so the whole component
    #: reports under ``sim.`` rather than beside observed counters — a reader
    #: putting two runs in one table must not be able to mistake one for the
    #: other.
    "sim": ("tokens", "lookups", "vram_hits", "ram_hits", "nvme_reads",
            "prefetched", "prefetch_used", "stalls", "seconds",
            "prefetch_seconds", "depth_changes"),
}


class Telemetry:
    """Writes events and snapshots for one run, in one schema.

    The run identifier and the host description are written once, as the
    opening event, so every later record can be short. A file with no opening
    event is a file from a crashed or truncated run, and ``read`` says so
    rather than returning records that cannot be attributed.
    """

    def __init__(self, path: str | os.PathLike | None = None, *,
                 run: str | None = None, clock=time.time) -> None:
        self.run = run or uuid.uuid4().hex[:12]
        self.clock = clock
        self.path = Path(path) if path else None
        self._fh = None
        self.records = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)   # line buffered

    # -- lifecycle ------------------------------------------------------

    def open_run(self, **fields: Any) -> Event:
        """The first record: who, where, and what this run is."""
        return self.event("run.open",
                          host=socket.gethostname(),
                          platform=platform.platform(),
                          python=platform.python_version(),
                          pid=os.getpid(),
                          **fields)

    def close_run(self, **fields: Any) -> Event:
        return self.event("run.close", **fields)

    # -- writing --------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> Event:
        e = Event(kind=kind, at=self.clock(), run=self.run, fields=dict(fields))
        self._write(e.to_json())
        return e

    def snapshot(self, sources: Mapping[str, Any] | None = None,
                 fields: Mapping[str, Iterable[str]] | None = None,
                 values: Mapping[str, Any] | None = None) -> Snapshot:
        """Read every named component's counters into one record.

        ``values`` takes an already-namespaced mapping, which is how an
        adapter contributes: a runtime's counters are normalised by the
        adapter that knows that runtime's shape, not by this module, which
        would otherwise have to know every runtime there is.
        """
        spec = fields or STANDARD_FIELDS
        sources = sources or {}
        collected = dict(values or {})
        for key in collected:
            if "." not in key:
                raise TelemetryError(
                    f"{key!r} is not namespaced; a bare counter cannot be told "
                    "apart from another component's")
        for namespace, source in sources.items():
            if source is None:
                continue
            names = spec.get(namespace)
            if names is None:
                raise TelemetryError(
                    f"no field list for namespace {namespace!r}; add it to "
                    "STANDARD_FIELDS rather than letting the schema drift")
            collected.update(collect(namespace, source, names))
        s = Snapshot(at=self.clock(), run=self.run, values=collected)
        self._write(s.to_json())
        return s

    def _write(self, line: str) -> None:
        self.records += 1
        if self._fh:
            self._fh.write(line + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Telemetry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# -- reading ------------------------------------------------------------


def read(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Every record in a telemetry file, in order, with its line checked."""
    out: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise TelemetryError(f"{path}:{lineno}: not a record ({e})") from None
            if rec.get("v") != SCHEMA_VERSION:
                raise TelemetryError(
                    f"{path}:{lineno}: schema version {rec.get('v')}, this reader "
                    f"knows {SCHEMA_VERSION}")
            if rec.get("type") not in ("event", "snapshot"):
                raise TelemetryError(f"{path}:{lineno}: unknown record type "
                                     f"{rec.get('type')!r}")
            out.append(rec)
    if not out:
        raise TelemetryError(f"{path}: no records")
    if out[0].get("kind") != "run.open":
        raise TelemetryError(
            f"{path}: does not open with run.open — a truncated or crashed run, "
            "whose records cannot be attributed to a host or a configuration")
    return out


def deltas(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Differences between consecutive snapshots.

    Counters here are cumulative, so a rate over an interval is the reader's
    job. This is that job done once, correctly: a key missing from either end
    is skipped rather than treated as zero.
    """
    previous: dict[str, Any] | None = None
    for rec in records:
        if rec.get("type") != "snapshot":
            continue
        values = rec["values"]
        if previous is not None:
            out = {"at": rec["at"], "seconds": rec["at"] - previous["at"]}
            for k, v in values.items():
                old = previous["values"].get(k)
                if isinstance(v, (int, float)) and isinstance(old, (int, float)):
                    out[k] = v - old
            yield out
        previous = rec

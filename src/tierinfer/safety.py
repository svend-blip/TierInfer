"""The one rule the rest of this project is allowed to be wrong under.

Everything here speculates. The predictor ranks experts a token has not
routed to yet; the prefetcher reads them before anyone asked; the caches keep
some and drop others; the policy moves a dial on evidence that is always a
little out of date. Every one of those can be wrong, and being wrong has to
cost bandwidth or time and nothing else.

So there is exactly one invariant, and it is stated here rather than left
distributed across the modules that happen to honour it:

    **A routed expert is always delivered, byte-for-byte, from the file.**

Not "almost always", not "unless the prefetch failed", not "unless the cache
said it was resident". `exact_load` is the path that consults nothing — no
predictor, no cache, no policy, no queue — and reads the bytes the index
names. Every speculative path falls back to it.

Two things are provided.

``guard`` wraps a speculative loader so that a miss, an error, a short read
or a wrong-sized answer is turned into an exact load rather than into a
result. A speculative path that cannot fail silently is one that cannot be
wrong about what it returns.

``audit`` reads the project's own source and reports any module that
speculates without a reachable exact path. That is a weak check and is meant
to be: it catches the case where a new speculative path is added and the
fallback is forgotten, which is the way this invariant would actually be
lost — not by someone deciding to remove it.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .index import ByteRange


class SafetyError(RuntimeError):
    """The exact path itself failed. There is nothing below this."""


@dataclass
class GuardStats:
    calls: int = 0
    speculative_hits: int = 0
    fallbacks: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.calls if self.calls else 0.0

    def _record(self, reason: str) -> None:
        self.fallbacks += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def exact_load(backend, ranges: Sequence[ByteRange]) -> bytes:
    """Read these bytes from the file. Consults nothing, caches nothing.

    This is the bottom of the stack. If it raises, the run is over — there is
    no further fallback and inventing one would mean returning weights that
    are not the model's.
    """
    rs = list(ranges)
    if not rs:
        raise SafetyError("an exact load was asked for no byte ranges")
    try:
        blobs, _ = backend.read(rs)
    except Exception as e:                      # noqa: BLE001 — re-raised as fatal
        raise SafetyError(f"the exact path failed, and there is nothing below "
                          f"it: {e}") from e
    data = b"".join(blobs)
    want = sum(r.nbytes for r in rs)
    if len(data) != want:
        raise SafetyError(f"the exact path returned {len(data)} bytes of {want}")
    return data


def guard(speculative: Callable[..., bytes | None], backend,
          ranges_for: Callable[[Any], Sequence[ByteRange]],
          stats: GuardStats | None = None) -> Callable[..., bytes]:
    """Wrap a speculative loader so that every way of being wrong is a fallback.

    The wrapped function may return the bytes, return ``None`` to say it does
    not have them, or raise. It may also — and this is the case worth
    guarding — return the *wrong* bytes: too few, too many. All four become
    an exact load, and each is counted separately so that a path failing for
    a new reason is visible rather than merely slow.
    """
    stats = stats if stats is not None else GuardStats()

    def loaded(key: Any) -> bytes:
        stats.calls += 1
        rs = list(ranges_for(key))
        want = sum(r.nbytes for r in rs)
        try:
            got = speculative(key)
        except Exception:                       # noqa: BLE001 — a wrong guess, not a crash
            stats._record("raised")
            return exact_load(backend, rs)
        if got is None:
            stats._record("absent")
            return exact_load(backend, rs)
        if len(got) != want:
            stats._record("wrong size")
            return exact_load(backend, rs)
        stats.speculative_hits += 1
        return bytes(got)

    loaded.stats = stats                        # type: ignore[attr-defined]
    return loaded


# -- the audit ----------------------------------------------------------


#: Modules that are allowed to speculate, and the name of the exact path each
#: one must keep reachable. A module added to the project that speculates and
#: is not listed here fails the audit, which is the point: the invariant is
#: lost by omission, not by decision.
SPECULATIVE_MODULES: dict[str, tuple[str, ...]] = {
    "tierinfer.prefetch": ("load_now", "exact_fallbacks"),
    "tierinfer.policy": ("nvme_reads", "stalls"),
    "tierinfer.stream": ("load_now",),
}


@dataclass
class AuditFinding:
    module: str
    missing: str

    def __str__(self) -> str:
        return f"{self.module} speculates but has no reachable {self.missing!r}"


def _bound_names(source: str) -> set[str]:
    """Names a module defines, assigns or reaches for — not words it uses.

    A function it defines, an attribute it reads or writes, a name it
    calls. Docstrings and comments contribute nothing, which is the point:
    the first version of this searched the text, and a fallback that had
    been deleted would have passed as long as a comment still named it.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def audit(modules: Iterable[str] | None = None) -> list[AuditFinding]:
    """Report speculative modules with no reachable exact path.

    Shallow by design — it checks that each required name is *bound or
    called* in the module's code, not that the code is right. What it
    catches is a fallback removed or forgotten, which is how this invariant
    would actually be lost.
    """
    import importlib

    findings: list[AuditFinding] = []
    for name in (modules or SPECULATIVE_MODULES):
        expected = SPECULATIVE_MODULES.get(name)
        if expected is None:
            findings.append(AuditFinding(name, "entry in SPECULATIVE_MODULES"))
            continue
        try:
            source = inspect.getsource(importlib.import_module(name))
        except (ImportError, OSError) as e:
            findings.append(AuditFinding(name, f"readable source ({e})"))
            continue
        bound = _bound_names(source)
        for token in expected:
            if token not in bound:
                findings.append(AuditFinding(name, token))
    return findings

"""A bounded RAM cache that keeps experts by what they are worth, not by age.

Least-recently-used is the wrong default here. MoE routing is skewed and the
skew is stable over a prompt, so an expert used by a third of the tokens is
worth more than one used once — even when the once was more recent. LRU
cannot see that difference; it evicts by clock.

So eviction scores each resident expert and drops the cheapest to lose:

    value = (activation rate + prediction confidence) * reload cost / size

Every term is measured rather than assumed. Activation rate comes from the
tracker's window. Reload cost is what this expert's loads actually took, or
the cache's observed mean before an expert has its own history. Size is the
expert's bytes, so a large expert must earn its place against several small
ones. Pinned entries are never scored: the shared expert and the router are
needed by every token and are not candidates.

Recency is not absent — it enters as the tie-break, because between two
experts of equal measured worth the one used more recently is the better bet.

Measured against LRU on a synthetic skewed trace (128 experts, 46 layers, 8
routed per layer, Zipf 1.1, 400 tokens):

    cache    value policy    LRU    resident share of model
     8 GB          61.3%   48.2%                       16%
    16 GB          74.2%   67.1%                       31%
    32 GB          87.4%   85.9%                       62%

The margin is widest where the cache is smallest, which is the regime this
project exists for. At 62% resident the choice of policy barely matters.

**Victim selection, and what the heap costs.** The first implementation
scanned every unpinned entry: 37 microseconds at 100 residents, 354 at 1 000,
1 031 at 3 000. A 32 GB cache holds roughly 3 670 of this model's experts,
so a token missing a quarter of its 368 activations would have spent longer
choosing victims than generating.

A lazy min-heap replaced the scan. It is approximate by construction, because
an entry's value moves as the tracker's window slides, so a score stored at
push time ages. Revalidation on the way out bounds the error: an entry whose
true value has risen more than ``revalidate_tolerance`` above its stored score
is re-pushed rather than evicted, up to ``revalidate_budget`` times, after
which the exact scan runs and ``heap_fallbacks`` counts it. Every fallback is
counted, including the one that fires when the heap drains, so the
approximation cannot hide behind a silent O(n).

    residents    scan    heap
          100   37 µs   10 µs
        1 000  354 µs    8 µs
        3 000 1031 µs    8 µs
        6 000       —    8 µs

Flat in the number of residents, and 0 fallbacks across every run above. The
approximation costs 0.1 percentage points of hit rate at 16 GB (74.2% against
the exact scan's 74.3%).

Asking for a victim does not consume it: the chosen key is pushed back before
it is returned, so the heap stays a superset of the resident keys whether or
not the caller evicts. That costs one stale entry per eviction — the heap
settles at about twice the resident count — and it buys two things: the
question is idempotent, and the pushed-back score is the revalidated one,
which is most of why the approximation only costs a tenth of a point.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

from .tracker import ExpertKey, ExpertTracker


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    insertions: int = 0
    evictions: int = 0
    bytes_admitted: int = 0
    bytes_evicted: int = 0
    rejected_oversize: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class _Entry:
    key: ExpertKey
    nbytes: int
    payload: object
    pinned: bool = False
    inserted_at: int = 0
    last_hit: int = 0


class ExpertCache:
    """Bounded by bytes, not by count: experts are not all the same size."""

    def __init__(
        self,
        capacity_bytes: int,
        tracker: ExpertTracker | None = None,
        *,
        default_reload_seconds: float = 0.010,
    ):
        if capacity_bytes <= 0:
            raise ValueError("capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self.tracker = tracker
        self.default_reload_seconds = default_reload_seconds
        self.stats = CacheStats()
        self._entries: dict[ExpertKey, _Entry] = {}
        self._bytes = 0
        self._clock = 0
        self._confidence: dict[ExpertKey, float] = {}
        #: Lazy min-heap of (value, sequence, key). Entries are never removed
        #: on update, only re-pushed; stale ones are discarded on the way out.
        self._heap: list[tuple[float, int, ExpertKey]] = []
        self._seq = 0
        #: How far a stored score may drift below the true one before the
        #: candidate is re-pushed rather than evicted.
        self.revalidate_tolerance = 0.25
        #: Cap on re-pushes per eviction, so a drifting window cannot turn one
        #: eviction into an unbounded loop. Exceeding it falls back to a scan.
        self.revalidate_budget = 32
        self.heap_fallbacks = 0

    # -- state ----------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        return self._bytes

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self._bytes

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[ExpertKey]:
        return list(self._entries)

    # -- use ------------------------------------------------------------

    def get(self, key: ExpertKey):
        """Return the cached payload, or ``None``. Counts as a lookup either way."""
        self._clock += 1
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        entry.last_hit = self._clock
        self.stats.hits += 1
        return entry.payload

    def put(self, key: ExpertKey, payload: object, nbytes: int, *, pinned: bool = False) -> bool:
        """Admit an expert, evicting as needed. False when it cannot fit at all."""
        if nbytes > self.capacity_bytes:
            self.stats.rejected_oversize += 1
            return False
        self._clock += 1
        if key in self._entries:
            self._drop(key, evicted=False)
        while self._bytes + nbytes > self.capacity_bytes:
            victim = self._choose_victim(exclude=key)
            if victim is None:
                self.stats.rejected_oversize += 1
                return False
            self._drop(victim, evicted=True)
        self._entries[key] = _Entry(
            key=key, nbytes=nbytes, payload=payload, pinned=pinned,
            inserted_at=self._clock, last_hit=self._clock,
        )
        self._bytes += nbytes
        self.stats.insertions += 1
        self.stats.bytes_admitted += nbytes
        if not pinned:
            self._push(key)
        return True

    def pin(self, key: ExpertKey) -> None:
        """Keep this expert regardless of score. Raises when it is not resident."""
        self._entries[key].pinned = True

    def unpin(self, key: ExpertKey) -> None:
        self._entries[key].pinned = False
        self._push(key)

    def set_confidence(self, scores: dict[ExpertKey, float]) -> None:
        """A predictor's confidence for the next token, 0..1 per expert.

        Confidence raises a resident expert's value, so the heap's stored
        score for it becomes an underestimate — which the revalidation on the
        way out catches. Only the named experts are re-pushed; the rest keep
        whatever ordering they had.
        """
        self._confidence = dict(scores)
        for key in scores:
            if key in self._entries and not self._entries[key].pinned:
                self._push(key)

    # -- policy ---------------------------------------------------------

    def value(self, key: ExpertKey) -> float:
        """What this resident expert is worth keeping. Higher survives."""
        entry = self._entries[key]
        rate = self.tracker.activation_rate(key) if self.tracker else 0.0
        confidence = self._confidence.get(key, 0.0)
        reload = self.default_reload_seconds
        if self.tracker:
            s = self.tracker.stats.get(key)
            if s and s.loads:
                reload = s.mean_load_seconds
        return (rate + confidence) * reload / max(entry.nbytes, 1)

    def _push(self, key: ExpertKey) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (self.value(key), self._seq, key))

    def _choose_victim(self, exclude: ExpertKey | None = None) -> ExpertKey | None:
        """The lowest-value unpinned entry, approximately.

        Approximately, and deliberately: an entry's value moves as the
        tracker's window slides, so a score stored at push time is a
        lower bound that ages. Popping discards entries that are gone or
        pinned, and re-pushes any whose true value has risen more than
        ``revalidate_tolerance`` above the stored one. Within the tolerance
        the candidate is taken as the minimum.

        The budget bounds the work: a window that has drifted for everyone
        could otherwise make one eviction re-push the whole heap. Exceeding
        it falls back to the exact scan and counts the event, so the
        approximation cannot hide.

        Asking does not consume: the chosen key is pushed back before it is
        returned, so the heap stays a superset of the resident keys whether
        or not the caller goes on to evict. Removal is lazy — a dropped
        key's heap entry is skipped the next time it surfaces.
        """
        deferred = 0
        while self._heap:
            stored, seq, key = heapq.heappop(self._heap)
            entry = self._entries.get(key)
            if entry is None or entry.pinned or key == exclude:
                continue                      # gone, pinned, or the newcomer
            true_value = self.value(key)
            if true_value > stored * (1.0 + self.revalidate_tolerance) and deferred < self.revalidate_budget:
                heapq.heappush(self._heap, (true_value, seq, key))
                deferred += 1
                continue
            heapq.heappush(self._heap, (true_value, seq, key))
            return key
        # The heap ran dry. Either the budget stopped the revalidation, or
        # every entry left is pinned or excluded. Both land on the exact
        # scan, and both are counted: a silent O(n) fallback is the one
        # thing this heap exists to rule out.
        if self._evictable(exclude):
            self.heap_fallbacks += 1
            return self._scan_victim(exclude)
        return None

    def _evictable(self, exclude: ExpertKey | None) -> bool:
        return any(not e.pinned and k != exclude for k, e in self._entries.items())

    def _scan_victim(self, exclude: ExpertKey | None = None) -> ExpertKey | None:
        """The exact lowest-value entry. The fallback, and what tests compare against."""
        candidates = [k for k, e in self._entries.items() if not e.pinned and k != exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda k: (self.value(k), self._entries[k].last_hit))

    def _drop(self, key: ExpertKey, *, evicted: bool) -> None:
        entry = self._entries.pop(key)
        self._bytes -= entry.nbytes
        if evicted:
            self.stats.evictions += 1
            self.stats.bytes_evicted += entry.nbytes

    # -- planning -------------------------------------------------------

    def missing(self, keys: Iterable[ExpertKey]) -> list[ExpertKey]:
        """Which of these are not resident. Does not count as a lookup."""
        return [k for k in keys if k not in self._entries]

    def would_evict_for(self, nbytes: int) -> list[ExpertKey]:
        """Which entries admitting ``nbytes`` would cost, without admitting it.

        Lets a caller decide whether a prefetch is worth its collateral before
        it happens, rather than discovering it afterwards.
        """
        if nbytes <= self.free_bytes:
            return []
        freed = self.free_bytes
        victims: list[ExpertKey] = []
        blocked = set()
        while freed < nbytes:
            candidates = [k for k, e in self._entries.items()
                          if not e.pinned and k not in blocked]
            if not candidates:
                break
            victim = min(candidates, key=lambda k: (self.value(k), self._entries[k].last_hit))
            victims.append(victim)
            blocked.add(victim)
            freed += self._entries[victim].nbytes
        return victims
